import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from use_croma import PretrainedCROMA
import torch.quantization as quant
import random
from tqdm import tqdm

NUM_CLASSES = 8
PATCH = 8

class DFCDataset(Dataset):
    def __init__(self, split="validation", n=None, seed=0):
        d = torch.load("./data/DFC_preprocessed.pt")
        imgs = d[f"{split}_images"]
        lbls = d[f"{split}_labels"]
        
        if n is not None and n < len(imgs):
            random.seed(seed)
            idx = random.sample(range(len(imgs)), n)
            imgs, lbls = imgs[idx], lbls[idx]
            
        self.images, self.labels = imgs, lbls
        
    def __len__(self): 
        return len(self.images)
        
    def __getitem__(self, i):
        img = self.images[i].float() / 255.0
        return img[12:], img[:12], self.labels[i].long() # Note: returned as s1, s2, lbl

def downsample_labels(labels, patch=PATCH):
    B, H, W = labels.shape
    ph, pw = H // patch, W // patch
    x = (
        labels.view(B, ph, patch, pw, patch)
        .permute(0, 1, 3, 2, 4)
        .reshape(B, ph * pw, patch * patch)
    )
    return torch.mode(x, dim=-1).values

val_ds = DFCDataset("validation", n=1000)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

print("1. Loading architecture and preparing for INT8 weight injection...")
croma = PretrainedCROMA(
    pretrained_path="CROMA_base.pt", 
    size="base", 
    modality="both", 
    image_resolution=96
).to("cpu")
croma.eval()

# We must prepare the model exactly as we did during conversion so the 
# state_dict keys match what is inside CROMA_INT8_Static.pt
croma.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(croma, inplace=True)
quant.convert(croma, inplace=True)

print("2. Loading Static INT8 weights...")
croma.load_state_dict(torch.load("CROMA_INT8_Static.pt"))

# Load linear probe for classification
probe = nn.Linear(768, NUM_CLASSES).to("cpu")
probe.load_state_dict(torch.load("probe_baseline.pt"))
probe.eval()

print("3. Evaluating Static Quantized mIoU...")
inter = torch.zeros(NUM_CLASSES)
union = torch.zeros(NUM_CLASSES)

with torch.no_grad():
    for s1, s2, lbl in tqdm(val_dl, desc="Running Inference"):
        out = croma(SAR_images=s1, optical_images=s2)
        feats = out["joint_encodings"]
        pred = probe(feats).argmax(-1)
        target = downsample_labels(lbl).clamp(min=0)
        for c in range(NUM_CLASSES):
            p_mask, t_mask = (pred == c), (target == c)
            inter[c] += (p_mask & t_mask).sum().item()
            union[c] += (p_mask | t_mask).sum().item()
miou = (inter / union.clamp(min=1)).mean().item()

print("4. Measuring Latency (Single Thread)...")
torch.set_num_threads(1)
dummy_s1 = torch.randn(1, 2, 96, 96)
dummy_s2 = torch.randn(1, 12, 96, 96)

for _ in range(5):
    _ = croma(SAR_images=dummy_s1, optical_images=dummy_s2)

t0 = time.perf_counter()
for _ in range(50):
    _ = croma(SAR_images=dummy_s1, optical_images=dummy_s2)
t1 = time.perf_counter()
latency_ms = ((t1 - t0) / 50) * 1000

model_size_mb = os.path.getsize("CROMA_INT8_Static.pt") / (1024 * 1024)

print("\n--- STATIC INT8 QUANTIZED RESULTS ---")
print(f"1. Static Quantized mIoU: {miou:.4f}")
print(f"2. Single-CPU Latency:    {latency_ms:.2f} ms/image")
print(f"3. Model File Size:       {model_size_mb:.2f} MB")

with open("results_static_quant.txt", "w") as f:
    f.write(f"{miou:.4f},{latency_ms:.2f},{model_size_mb:.2f}")