import os, time, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pruned_arch import build_structural_pruned_croma
from tqdm import tqdm

NUM_CLASSES = 8
PATCH = 8

class DFCDataset(Dataset):
    def __init__(self, split="validation_images", n=None, seed=0):
        d = torch.load("./data/DFC_preprocessed.pt")
        imgs, lbls = d[split], d[split.replace("images", "labels")]
        if n is not None and n < len(imgs):
            random.seed(seed)
            idx = random.sample(range(len(imgs)), n)
            imgs, lbls = imgs[idx], lbls[idx]
        self.images, self.labels = imgs, lbls
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = self.images[i].float() / 255.0
        return img[12:], img[:12], self.labels[i].long()

def downsample_labels(labels, patch=PATCH):
    B, H, W = labels.shape
    ph, pw = H // patch, W // patch
    x = labels.view(B, ph, patch, pw, patch).permute(0,1,3,2,4).reshape(B, ph*pw, patch*patch)
    return torch.mode(x, dim=-1).values

def main():
    val_ds = DFCDataset("validation_images", n=1000)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

    print("Rebuilding structural-pruned architecture...")
    croma_structural = build_structural_pruned_croma()

    print("Quantizing the structurally pruned model...")
    croma_hybrid = torch.quantization.quantize_dynamic(croma_structural, {nn.Linear}, dtype=torch.qint8)
    torch.save(croma_hybrid.state_dict(), "croma_hybrid_structural.pt")

    probe = nn.Linear(768, NUM_CLASSES).to("cpu")
    probe.load_state_dict(torch.load("probe_baseline.pt"))
    probe.eval()

    inter = torch.zeros(NUM_CLASSES); union = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for s1, s2, lbl in tqdm(val_dl, desc="Eval"):
            out = croma_hybrid(SAR_images=s1, optical_images=s2)
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
        for _ in range(5): croma_hybrid(SAR_images=dummy_s1, optical_images=dummy_s2)
        t0 = time.perf_counter()
        for _ in range(50): croma_hybrid(SAR_images=dummy_s1, optical_images=dummy_s2)
        t1 = time.perf_counter()
    latency_ms = (t1-t0)/50*1000
    size_mb = os.path.getsize("croma_hybrid_structural.pt") / (1024*1024)

    print(f"\n--- HYBRID (STRUCTURAL + INT8) RESULTS ---")
    print(f"mIoU: {miou:.4f}  |  Latency: {latency_ms:.2f} ms/img  |  Size: {size_mb:.2f} MB")
    with open("results_hybrid_structural.txt", "w") as f:
        f.write(f"{miou:.4f},{latency_ms:.2f},{size_mb:.2f}")

if __name__ == "__main__":
    main()