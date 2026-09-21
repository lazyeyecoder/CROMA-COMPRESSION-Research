import torch, torch.nn as nn, torch.nn.functional as F

NUM_CLASSES = 8
PATCH = 8

def downsample_labels(labels, patch=PATCH):
    B, H, W = labels.shape
    ph, pw = H // patch, W // patch
    x = labels.view(B, ph, patch, pw, patch).permute(0, 1, 3, 2, 4).reshape(B, ph*pw, patch*patch)
    return torch.mode(x, dim=-1).values.clamp(min=0)

train = torch.load("dfc_train_features.pt")
val   = torch.load("dfc_val_features.pt")

train_targets = downsample_labels(train["lbls"])   # (N,144) — cheap, no model needed
val_targets   = downsample_labels(val["lbls"])

probe = nn.Linear(768, NUM_CLASSES)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

ds = torch.utils.data.TensorDataset(train["feats"], train_targets)
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

for epoch in range(30):   # cheap now, can afford more epochs
    for feats, targets in dl:
        logits = probe(feats)
        loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), targets.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch} loss {loss.item():.4f}")

# ---- mIoU on val ----
probe.eval()
with torch.no_grad():
    pred = probe(val["feats"]).argmax(-1)
inter = torch.zeros(NUM_CLASSES); union = torch.zeros(NUM_CLASSES)
for c in range(NUM_CLASSES):
    p, t = (pred == c), (val_targets == c)
    inter[c] += (p & t).sum().item()
    union[c] += (p | t).sum().item()
iou = inter / union.clamp(min=1)
print(f"BASELINE mIoU: {iou.mean().item():.4f}")
torch.save(probe.state_dict(), "probe_baseline.pt")