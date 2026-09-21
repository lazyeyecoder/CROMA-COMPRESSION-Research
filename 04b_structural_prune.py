import os, time, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from einops import einsum
from use_croma import PretrainedCROMA
from tqdm import tqdm

NUM_CLASSES = 8
PATCH = 8
NUM_HEADS = 16
HEAD_DIM = 48
PRUNE_RATIO = 0.2
N_KEEP = NUM_HEADS - int(NUM_HEADS * PRUNE_RATIO)  # 13

class DFCDataset(Dataset):
    def __init__(self, split="validation", n=None, seed=0):
        d = torch.load("./data/DFC_preprocessed.pt")
        imgs, lbls = d[f"{split}_images"], d[f"{split}_labels"]
        if n is not None and n < len(imgs):
            random.seed(seed)
            idx = random.sample(range(len(imgs)), n)
            imgs, lbls = imgs[idx], lbls[idx]
        self.images, self.labels = imgs, lbls
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = self.images[i].float() / 255.0
        return img[12:], img[:12], self.labels[i].long()  # s1, s2, lbl

def downsample_labels(labels, patch=PATCH):
    B, H, W = labels.shape
    ph, pw = H // patch, W // patch
    x = labels.view(B, ph, patch, pw, patch).permute(0,1,3,2,4).reshape(B, ph*pw, patch*patch)
    return torch.mode(x, dim=-1).values

# ---- small custom module, output dim = N_KEEP*HEAD_DIM instead of full dim, to_out projects back to 768 ----
class PrunedAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, num_heads*head_dim*3, bias=False)
        self.to_out = nn.Linear(num_heads*head_dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, relative_position_bias):
        x = self.input_norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        scores = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        scores = scores + relative_position_bias
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PrunedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(dim, num_heads*head_dim, bias=False)
        self.to_k = nn.Linear(dim, num_heads*head_dim, bias=False)
        self.to_v = nn.Linear(dim, num_heads*head_dim, bias=False)
        self.to_out = nn.Linear(num_heads*head_dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, context, relative_position_bias):
        x = self.input_norm(x)
        context = self.input_norm(context)
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        scores = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        scores = scores + relative_position_bias
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def head_importance_self(attn, dim=768, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
    w = attn.to_qkv.weight.data
    scores = torch.zeros(num_heads)
    for h in range(num_heads):
        s = slice(h*head_dim, (h+1)*head_dim)
        q, k, v = w[s], w[dim+s.start:dim+s.stop], w[2*dim+s.start:2*dim+s.stop]
        scores[h] = q.abs().sum() + k.abs().sum() + v.abs().sum()
    return scores

def head_importance_cross(cattn, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
    wq, wk, wv = cattn.to_q.weight.data, cattn.to_k.weight.data, cattn.to_v.weight.data
    scores = torch.zeros(num_heads)
    for h in range(num_heads):
        s = slice(h*head_dim, (h+1)*head_dim)
        scores[h] = wq[s].abs().sum() + wk[s].abs().sum() + wv[s].abs().sum()
    return scores

def build_pruned_self_attn(old_attn, kept_idx, dim=768, head_dim=HEAD_DIM):
    n = len(kept_idx)
    new_attn = PrunedAttention(dim, n, head_dim)
    w = old_attn.to_qkv.weight.data
    rows = []
    for h in kept_idx: rows.append(w[h*head_dim:(h+1)*head_dim])
    for h in kept_idx: rows.append(w[dim+h*head_dim:dim+(h+1)*head_dim])
    for h in kept_idx: rows.append(w[2*dim+h*head_dim:2*dim+(h+1)*head_dim])
    new_attn.to_qkv.weight.data = torch.cat(rows, dim=0)
    cols = [old_attn.to_out.weight.data[:, h*head_dim:(h+1)*head_dim] for h in kept_idx]
    new_attn.to_out.weight.data = torch.cat(cols, dim=1)
    new_attn.to_out.bias.data = old_attn.to_out.bias.data.clone()
    new_attn.input_norm.load_state_dict(old_attn.input_norm.state_dict())
    return new_attn

def build_pruned_cross_attn(old_cattn, kept_idx, dim=768, head_dim=HEAD_DIM):
    n = len(kept_idx)
    new_cattn = PrunedCrossAttention(dim, n, head_dim)
    for name in ["to_q", "to_k", "to_v"]:
        old_w = getattr(old_cattn, name).weight.data
        rows = [old_w[h*head_dim:(h+1)*head_dim] for h in kept_idx]
        getattr(new_cattn, name).weight.data = torch.cat(rows, dim=0)
    cols = [old_cattn.to_out.weight.data[:, h*head_dim:(h+1)*head_dim] for h in kept_idx]
    new_cattn.to_out.weight.data = torch.cat(cols, dim=1)
    new_cattn.to_out.bias.data = old_cattn.to_out.bias.data.clone()
    new_cattn.input_norm.load_state_dict(old_cattn.input_norm.state_dict())
    return new_cattn

def main():
    print("Loading CROMA...")
    croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96).to("cpu")
    croma.eval()

    # ---- 1. gather ALL self-attn and cross-attn modules across the whole model ----
    self_modules = []
    for self_attn, ffn in croma.s2_encoder.transformer.layers: self_modules.append(self_attn)
    for self_attn, ffn in croma.s1_encoder.transformer.layers: self_modules.append(self_attn)
    cross_self_modules, cross_cross_modules = [], []
    for self_attn, cross_attn, ffn in croma.cross_encoder.layers:
        cross_self_modules.append(self_attn)
        cross_cross_modules.append(cross_attn)
    self_modules += cross_self_modules

    # ---- 2. one combined global importance score across every attention module in the model ----
    total_score = torch.zeros(NUM_HEADS)
    for m in self_modules: total_score += head_importance_self(m)
    for m in cross_cross_modules: total_score += head_importance_cross(m)

    kept_idx = total_score.argsort(descending=True)[:N_KEEP].sort().values.tolist()
    print(f"Keeping heads (global): {kept_idx}")

    # ---- 3. rebuild every attention module with only kept heads ----
    print("Rebuilding s2_encoder...")
    for i, (self_attn, ffn) in enumerate(croma.s2_encoder.transformer.layers):
        croma.s2_encoder.transformer.layers[i][0] = build_pruned_self_attn(self_attn, kept_idx)
    print("Rebuilding s1_encoder...")
    for i, (self_attn, ffn) in enumerate(croma.s1_encoder.transformer.layers):
        croma.s1_encoder.transformer.layers[i][0] = build_pruned_self_attn(self_attn, kept_idx)
    print("Rebuilding cross_encoder...")
    for i, (self_attn, cross_attn, ffn) in enumerate(croma.cross_encoder.layers):
        croma.cross_encoder.layers[i][0] = build_pruned_self_attn(self_attn, kept_idx)
        croma.cross_encoder.layers[i][1] = build_pruned_cross_attn(cross_attn, kept_idx)

    # ---- 4. slice the shared ALiBi bias to match (this is what makes it all consistent) ----
    croma.attn_bias = croma.attn_bias[:, kept_idx, :, :]

    torch.save(croma.state_dict(), "croma_structural_pruned.pt")
    size_mb = os.path.getsize("croma_structural_pruned.pt") / (1024*1024)
    print(f"Saved. Real size: {size_mb:.2f} MB")

    # ---- 5. eval ----
    probe = nn.Linear(768, NUM_CLASSES).to("cpu")
    probe.load_state_dict(torch.load("probe_baseline.pt"))
    probe.eval()

    val_dl = DataLoader(DFCDataset("validation", n=1000), batch_size=16, shuffle=False)
    inter = torch.zeros(NUM_CLASSES); union = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for s1, s2, lbl in tqdm(val_dl, desc="Eval"):
            out = croma(SAR_images=s1, optical_images=s2)
            pred = probe(out["joint_encodings"]).argmax(-1)
            target = downsample_labels(lbl).clamp(min=0)
            for c in range(NUM_CLASSES):
                p, t = (pred == c), (target == c)
                inter[c] += (p & t).sum().item()
                union[c] += (p | t).sum().item()
    miou = (inter / union.clamp(min=1)).mean().item()

    torch.set_num_threads(1)
    dummy_s1, dummy_s2 = torch.randn(1,2,96,96), torch.randn(1,12,96,96)
    with torch.no_grad():
        for _ in range(5): croma(SAR_images=dummy_s1, optical_images=dummy_s2)
        t0 = time.perf_counter()
        for _ in range(50): croma(SAR_images=dummy_s1, optical_images=dummy_s2)
        t1 = time.perf_counter()
    latency_ms = (t1-t0)/50*1000

    print(f"\n--- STRUCTURAL PRUNED (13/16 heads, real matrix shrink) ---")
    print(f"mIoU: {miou:.4f}  |  Latency: {latency_ms:.2f} ms/img  |  Size: {size_mb:.2f} MB")
    with open("results_structural_pruned.txt", "w") as f:
        f.write(f"{miou:.4f},{latency_ms:.2f},{size_mb:.2f}")

if __name__ == "__main__":
    main()