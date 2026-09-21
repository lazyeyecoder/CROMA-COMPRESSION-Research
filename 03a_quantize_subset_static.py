import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from use_croma import PretrainedCROMA
import torch.quantization as quant
from tqdm import tqdm  # ADDED TQDM

device = "cpu"

print("1. Loading FP32 Baseline CROMA...")
croma = PretrainedCROMA(
    pretrained_path="CROMA_base.pt", 
    size="base", 
    modality="both", 
    image_resolution=96
).to(device)
croma.eval()

# To quantize properly, we must define a QConfig.
# fbgemm is standard for x86 CPUs, qnnpack is standard for ARM (like Jetson/Satellites).
# Since you are simulating on x86, we use fbgemm for now.
croma.qconfig = quant.get_default_qconfig('fbgemm')

print("2. Preparing model for Post-Training Static Quantization...")
# prepare() inserts observers that watch the tensors during the calibration run
quant.prepare(croma, inplace=True)

print("3. Setting up Calibration Dataset...")
class DFCDataset(Dataset):
    def __init__(self, split="train", n=None, seed=0):
        d = torch.load("./data/DFC_preprocessed.pt")
        imgs = d[f"{split}_images"]
        lbls = d[f"{split}_labels"]
        
        if n is not None and n < len(imgs):
            import random
            random.seed(seed)
            idx = random.sample(range(len(imgs)), n)
            imgs, lbls = imgs[idx], lbls[idx]
            
        self.images, self.labels = imgs, lbls
        
    def __len__(self): 
        return len(self.images)
        
    def __getitem__(self, i):
        img = self.images[i].float() / 255.0
        return img[:12], img[12:], self.labels[i].long()

# Using 32 images for calibration
calib_loader = DataLoader(DFCDataset("train", n=32), batch_size=8, num_workers=0)

print("4. Running Calibration (feeding 32 images so observers can calculate INT8 bounds)...")
with torch.no_grad():
    # ADDED TQDM HERE
    for s2, s1, _ in tqdm(calib_loader, desc="Calibrating"): 
        croma(SAR_images=s1, optical_images=s2)

print("\n5. Converting observed FP32 weights to INT8...")
quant.convert(croma, inplace=True)

print("6. Saving Quantized Model...")
torch.save(croma.state_dict(), "CROMA_INT8_Static.pt")

model_size_mb = os.path.getsize("CROMA_INT8_Static.pt") / (1024 * 1024)
print(f"SUCCESS! CROMA is now an INT8 Static quantized model.")
print(f"New Model File Size: {model_size_mb:.2f} MB")