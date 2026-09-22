import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

NUM_CLASSES = 8
PATCH = 8

def downsample_labels(labels, patch=PATCH):
    B, H, W = labels.shape
    ph, pw = H // patch, W // patch
    x = labels.view(B, ph, patch, pw, patch).permute(0, 1, 3, 2, 4).reshape(B, ph*pw, patch*patch)
    return torch.mode(x, dim=-1).values.clamp(min=0)

train = torch.load("dfc_train_features_full.pt", mmap=True)
val   = torch.load("dfc_val_features_full.pt", mmap=True)

print("train samples:", train["feats"].shape[0])   # sanity check, expect ~46152
print("val samples:", val["feats"].shape[0])        # expect ~8874

train_targets = downsample_labels(train["lbls"])
val_targets   = downsample_labels(val["lbls"])

probe = nn.Linear(768, NUM_CLASSES)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

torch.set_num_threads(16)

train_ds = TensorDataset(train["feats"], train_targets)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

val_ds = TensorDataset(val["feats"], val_targets)
val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

print("Starting training...")
for epoch in range(30):
    probe.train()
    total_loss = 0.0
    for feats, targets in train_dl:
        logits = probe(feats)
        loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if epoch % 5 == 0 or epoch == 29:
        print(f"epoch {epoch} | avg loss: {total_loss / len(train_dl):.4f}")

print("Starting validation...")
probe.eval()
inter = torch.zeros(NUM_CLASSES)
union = torch.zeros(NUM_CLASSES)

with torch.no_grad():
    for feats, targets in val_dl:
        preds = probe(feats).argmax(-1)
        for c in range(NUM_CLASSES):
            p, t = (preds == c), (targets == c)
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()

present_classes = union > 0
iou = inter[present_classes] / union[present_classes]
print(f"per-class IoU (present classes only): {iou}")
print(f"classes present: {present_classes.sum().item()} / {NUM_CLASSES}")
print(f"BASELINE mIoU: {iou.mean().item():.4f}")

torch.save(probe.state_dict(), "probe_baseline_full.pt")

with open("results_baseline_full.txt", "w") as f:
    f.write(f"{iou.mean().item():.4f}")