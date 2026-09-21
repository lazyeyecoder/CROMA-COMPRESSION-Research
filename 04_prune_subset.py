import os, time, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from use_croma import PretrainedCROMA
from tqdm import tqdm

NUM_CLASSES = 8
PATCH = 8
NUM_HEADS = 16
PRUNE_RATIO = 0.2

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
    x = labels.view(B, ph, patch, pw, patch).permute(0, 1, 3, 2, 4).reshape(B, ph*pw, patch*patch)
    return torch.mode(x, dim=-1).values

# ---- self-attention pruning (s1_encoder, s2_encoder, and self_attn inside cross_encoder) ----
def prune_self_attn(attn, prune_ratio=PRUNE_RATIO, num_heads=NUM_HEADS):
    w = attn.to_qkv.weight.data          # (2304, 768) = [q;k;v] stacked
    dim = w.shape[1]                     # 768
    head_dim = dim // num_heads          # 48
    scores = []
    for h in range(num_heads):
        sl = slice(h*head_dim, (h+1)*head_dim)
        q, k, v = w[sl], w[dim+sl.start:dim+sl.stop], w[2*dim+sl.start:2*dim+sl.stop]
        scores.append((q.abs().sum() + k.abs().sum() + v.abs().sum()).item())
    scores = torch.tensor(scores)
    prune_idx = scores.argsort()[:int(num_heads*prune_ratio)]
    for h in prune_idx:
        sl = slice(h*head_dim, (h+1)*head_dim)
        w[sl] = 0
        w[dim+sl.start:dim+sl.stop] = 0
        w[2*dim+sl.start:2*dim+sl.stop] = 0

# ---- cross-attention pruning (separate q/k/v matrices) ----
def prune_cross_attn(cross_attn, prune_ratio=PRUNE_RATIO, num_heads=NUM_HEADS):
    wq, wk, wv = cross_attn.to_q.weight.data, cross_attn.to_k.weight.data, cross_attn.to_v.weight.data
    dim = wq.shape[0]
    head_dim = dim // num_heads
    scores = []
    for h in range(num_heads):
        sl = slice(h*head_dim, (h+1)*head_dim)
        scores.append((wq[sl].abs().sum() + wk[sl].abs().sum() + wv[sl].abs().sum()).item())
    scores = torch.tensor(scores)
    prune_idx = scores.argsort()[:int(num_heads*prune_ratio)]
    for h in prune_idx:
        sl = slice(h*head_dim, (h+1)*head_dim)
        wq[sl] = 0; wk[sl] = 0; wv[sl] = 0

def main():
    print("Loading CROMA...")
    croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96).to("cpu")
    croma.eval()

    print("Pruning s2_encoder (self-attn only)...")
    for self_attn, ffn in croma.s2_encoder.transformer.layers:
        prune_self_attn(self_attn)

    print("Pruning s1_encoder (self-attn only)...")
    for self_attn, ffn in croma.s1_encoder.transformer.layers:
        prune_self_attn(self_attn)

    print("Pruning cross_encoder (self-attn + cross-attn)...")
    for self_attn, cross_attn, ffn in croma.cross_encoder.layers:
        prune_self_attn(self_attn)
        prune_cross_attn(cross_attn)

    torch.save(croma.state_dict(), "croma_pruned_20.pt")

    probe = nn.Linear(768, NUM_CLASSES).to("cpu")
    probe.load_state_dict(torch.load("probe_baseline.pt"))
    probe.eval()

    val_dl = DataLoader(DFCDataset("validation", n=1000), batch_size=16, shuffle=False)
    inter = torch.zeros(NUM_CLASSES); union = torch.zeros(NUM_CLASSES)
    print("Evaluating pruned model...")
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

    print("Measuring latency...")
    torch.set_num_threads(1)
    dummy_s1, dummy_s2 = torch.randn(1,2,96,96), torch.randn(1,12,96,96)
    with torch.no_grad():
        for _ in range(5): croma(SAR_images=dummy_s1, optical_images=dummy_s2)
        t0 = time.perf_counter()
        for _ in range(50): croma(SAR_images=dummy_s1, optical_images=dummy_s2)
        t1 = time.perf_counter()
    latency_ms = (t1-t0)/50*1000
    size_mb = os.path.getsize("croma_pruned_20.pt") / (1024*1024)

    print(f"\n--- PRUNED 20% (self-attn + cross-attn) RESULTS ---")
    print(f"mIoU: {miou:.4f}  |  Latency: {latency_ms:.2f} ms/img  |  Size: {size_mb:.2f} MB")

    with open("results_pruned20.txt", "w") as f:
        f.write(f"{miou:.4f},{latency_ms:.2f},{size_mb:.2f}")

if __name__ == "__main__":
    main()