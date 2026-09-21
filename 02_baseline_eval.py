import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from use_croma import PretrainedCROMA

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 8
PATCH = 8


# 1. Dataset Loader
class DFCDataset(Dataset):

  def __init__(self, split="train_images"):
    d = torch.load("./data/DFC_preprocessed.pt")
    self.images = d[split]  # (N,14,96,96) uint8
    self.labels = d[split.replace("images", "labels")]  # (N,96,96)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, i):
    img = self.images[i].float() / 255.0  # 0-1 range
    s2 = img[:12]  # First 12 = Sentinel-2 Optical
    s1 = img[12:]  # Last 2 = Sentinel-1 SAR
    lbl = self.labels[i].long()
    return s1, s2, lbl


def downsample_labels(labels, patch=PATCH):
  B, H, W = labels.shape
  ph, pw = H // patch, W // patch
  x = (
      labels.view(B, ph, patch, pw, patch)
      .permute(0, 1, 3, 2, 4)
      .reshape(B, ph * pw, patch * patch)
  )
  return torch.mode(x, dim=-1).values


train_ds = DFCDataset("train_images")
val_ds = DFCDataset("validation_images")
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

# 2. Load Pretrained CROMA Backbone
print("Loading CROMA Base Model...")
croma = PretrainedCROMA(
    pretrained_path="CROMA_base.pt",
    size="base",
    modality="both",
    image_resolution=96,
).to(device)
croma.eval()
for p in croma.parameters():
  p.requires_grad = False

probe = nn.Linear(768, NUM_CLASSES).to(device)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

# 3. Train Probe
print("Training Linear Probe on DFC2020...")
for epoch in range(5):
  probe.train()
  for s1, s2, lbl in train_dl:
    s1, s2, lbl = s1.to(device), s2.to(device), lbl.to(device)
    with torch.no_grad():
      out = croma(SAR_images=s1, optical_images=s2)
      feats = out["joint_encodings"]  # (B, 144, 768)
    logits = probe(feats)
    target = downsample_labels(lbl).to(device).clamp(min=0)
    loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), target.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
  print(f" Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

# 4. Measure mIoU
probe.eval()
inter = torch.zeros(NUM_CLASSES)
union = torch.zeros(NUM_CLASSES)
with torch.no_grad():
  for s1, s2, lbl in val_dl:
    s1, s2, lbl = s1.to(device), s2.to(device), lbl.to(device)
    out = croma(SAR_images=s1, optical_images=s2)
    feats = out["joint_encodings"]
    pred = probe(feats).argmax(-1)
    target = downsample_labels(lbl).clamp(min=0)
    for c in range(NUM_CLASSES):
      p_mask, t_mask = (pred == c), (target == c)
      inter[c] += (p_mask & t_mask).sum().item()
      union[c] += (p_mask | t_mask).sum().item()
miou = (inter / union.clamp(min=1)).mean().item()

# 5. Measure Latency on Single-Thread CPU
torch.set_num_threads(1)
croma_cpu = croma.to("cpu")
dummy_s1 = torch.randn(1, 2, 96, 96)
dummy_s2 = torch.randn(1, 12, 96, 96)

# Warmup
for _ in range(5):
  _ = croma_cpu(SAR_images=dummy_s1, optical_images=dummy_s2)

t0 = time.perf_counter()
for _ in range(50):
  _ = croma_cpu(SAR_images=dummy_s1, optical_images=dummy_s2)
t1 = time.perf_counter()
latency_ms = ((t1 - t0) / 50) * 1000
model_size_mb = os.path.getsize("CROMA_base.pt") / (1024 * 1024)

torch.save(probe.state_dict(), "probe_baseline.pt")

print("\n--- FP32 BASELINE RESULTS ---")
print(f"1. Validation mIoU:    {miou:.4f}")
print(f"2. Single-CPU Latency: {latency_ms:.2f} ms/image")
print(f"3. Model File Size:    {model_size_mb:.2f} MB")

# Save results locally for Step 6
with open("results_baseline.txt", "w") as f:
  f.write(f"{miou:.4f},{latency_ms:.2f},{model_size_mb:.2f}")