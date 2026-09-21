import torch, torch.nn as nn
from einops import rearrange, einsum
from use_croma import PretrainedCROMA

NUM_HEADS = 16
HEAD_DIM = 48
PRUNE_RATIO = 0.2
N_KEEP = NUM_HEADS - int(NUM_HEADS * PRUNE_RATIO)

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
    rows = [w[h*head_dim:(h+1)*head_dim] for h in kept_idx]
    rows += [w[dim+h*head_dim:dim+(h+1)*head_dim] for h in kept_idx]
    rows += [w[2*dim+h*head_dim:2*dim+(h+1)*head_dim] for h in kept_idx]
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

def build_structural_pruned_croma(pretrained_path="CROMA_base.pt"):
    """Rebuilds the exact same 13/16-head architecture as 04b, then loads the saved weights into it."""
    croma = PretrainedCROMA(pretrained_path=pretrained_path, size="base", modality="both", image_resolution=96).to("cpu")
    croma.eval()

    self_modules = []
    for self_attn, ffn in croma.s2_encoder.transformer.layers: self_modules.append(self_attn)
    for self_attn, ffn in croma.s1_encoder.transformer.layers: self_modules.append(self_attn)
    cross_self_modules, cross_cross_modules = [], []
    for self_attn, cross_attn, ffn in croma.cross_encoder.layers:
        cross_self_modules.append(self_attn)
        cross_cross_modules.append(cross_attn)
    self_modules += cross_self_modules

    total_score = torch.zeros(NUM_HEADS)
    for m in self_modules: total_score += head_importance_self(m)
    for m in cross_cross_modules: total_score += head_importance_cross(m)
    kept_idx = total_score.argsort(descending=True)[:N_KEEP].sort().values.tolist()

    for i, (self_attn, ffn) in enumerate(croma.s2_encoder.transformer.layers):
        croma.s2_encoder.transformer.layers[i][0] = build_pruned_self_attn(self_attn, kept_idx)
    for i, (self_attn, ffn) in enumerate(croma.s1_encoder.transformer.layers):
        croma.s1_encoder.transformer.layers[i][0] = build_pruned_self_attn(self_attn, kept_idx)
    for i, (self_attn, cross_attn, ffn) in enumerate(croma.cross_encoder.layers):
        croma.cross_encoder.layers[i][0] = build_pruned_self_attn(self_attn, kept_idx)
        croma.cross_encoder.layers[i][1] = build_pruned_cross_attn(cross_attn, kept_idx)

    croma.attn_bias = croma.attn_bias[:, kept_idx, :, :]

    # load the exact saved weights on top (redundant given determinism, but safe/explicit)
    state = torch.load("croma_structural_pruned.pt")
    croma.load_state_dict(state)
    return croma