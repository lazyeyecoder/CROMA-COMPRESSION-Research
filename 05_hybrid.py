import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from use_croma import PretrainedCROMA

NUM_CLASSES = 8
PATCH = 8


class DFCDataset(Dataset):

  def __init__(self, split="validation_images"):
    d = torch.load("./data/DFC_preprocessed.pt")
    self.images = d[split]
    self.labels = d[split.replace("images", "labels")]

  def __len__(self):
    return len(self.images)

  def __getitem__(self, i):
    img = self.images[i].float() / 255.0
    s2, s1 = img[:12], img[12:]
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


val_ds = DFCDataset("validation_images")
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

print("Applying Hybrid Compression (Pruned 20% + INT8 Quantization)...")
croma_pruned = PretrainedCROMA(
    pretrained_path="CROMA_base.pt",
    size="base",
    modality="both",
    image_resolution=96,
).to("cpu")
croma_pruned.load_state_dict(torch.load("croma_pruned_20.pt"))
croma_pruned.eval()

# Quantize the pruned model
croma_hybrid = torch.quantization.quantize_dynamic(
    croma_pruned, {nn.Linear}, dtype=torch.qint8
)
torch.save(croma_hybrid.state_dict(), "croma_hybrid.pt")

probe = nn.Linear(768, NUM_CLASSES).to("cpu")
probe.load_state_dict(torch.load("probe_baseline.pt"))
probe.eval()

# Evaluate Hybrid mIoU
inter = torch.zeros(NUM_CLASSES)
union = torch.zeros(NUM_CLASSES)
with torch.no_grad():
  for s1, s2, lbl in val_dl:
    out = croma_hybrid(SAR_images=s1, optical_images=s2)
    feats = out["joint_encodings"]
    pred = probe(feats).argmax(-1)
    target = downsample_labels(lbl).clamp(min=0)
    for c in range(NUM_CLASSES):
      p_mask, t_mask = (pred == c), (target == c)
      inter[c] += (p_mask & t_mask).sum().item()
      union[c] += (p_mask | t_mask).sum().item()
miou = (inter / union.clamp(min=1)).mean().item()

torch.set_num_threads(1)
dummy_s1 = torch.randn(1, 2, 96, 96)
dummy_s2 = torch.randn(1, 12, 96, 96)

for _ in range(5):
  _ = croma_hybrid(SAR_images=dummy_s1, optical_images=dummy_s2)

t0 = time.perf_counter()
for _ in range(50):
  _ = croma_hybrid(SAR_images=dummy_s1, optical_images=dummy_s2)
t1 = time.perf_counter()
latency_ms = ((t1 - t0) / 50) * 1000
model_size_mb = os.path.getsize("croma_hybrid.pt") / (1024 * 1024)

print("\n--- HYBRID (PRUNED + INT8) RESULTS ---")
print(f"1. Hybrid mIoU:        {miou:.4f}")
print(f"2. Single-CPU Latency: {latency_ms:.2f} ms/image")
print(f"3. Model File Size:    {model_size_mb:.2f} MB")

with open("results_hybrid.txt", "w") as f:
  f.write(f"{miou:.4f},{latency_ms:.2f},{model_size_mb:.2f}")