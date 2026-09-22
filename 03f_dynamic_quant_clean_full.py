import os, time, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from use_croma import PretrainedCROMA
from tqdm import tqdm

NUM_CLASSES = 8
PATCH = 8
device = "cpu"

class DFCDataset(Dataset):
    def __init__(self, split, n=None, seed=0):
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
        return img[:12], img[12:], self.labels[i].long()

def downsample_labels(labels, patch=PATCH):
    B, H, W = labels.shape
    ph, pw = H // patch, W // patch
    x = labels.view(B, ph, patch, pw, patch).permute(0, 1, 3, 2, 4).reshape(B, ph*pw, patch*patch)
    return torch.mode(x, dim=-1).values.clamp(min=0)

def main():
    croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96).to(device)
    croma.eval()

    croma_q = torch.quantization.quantize_dynamic(croma, {nn.Linear}, dtype=torch.qint8)
    torch.save(croma_q.state_dict(), "CROMA_INT8_Dynamic_full.pt")
    size_mb = os.path.getsize("CROMA_INT8_Dynamic_full.pt") / (1024*1024)
    print(f"dynamic quantized, size = {size_mb:.2f} MB")

    torch.set_num_threads(16)   # use the 701 PC's cores for the eval forward-pass loop, same as probe training did
    val_dl = DataLoader(DFCDataset("validation", n=None), batch_size=16, num_workers=0)   # n=None = FULL 8,874 val images
    probe = nn.Linear(768, NUM_CLASSES).to(device)
    probe.load_state_dict(torch.load("probe_baseline_full.pt"))   # the FULL-data probe, not the subset one
    probe.eval()

    inter = torch.zeros(NUM_CLASSES); union = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for s2, s1, lbl in tqdm(val_dl, desc="Eval mIoU (full)"):
            out = croma_q(SAR_images=s1, optical_images=s2)
            pred = probe(out["joint_encodings"]).argmax(-1)
            target = downsample_labels(lbl)
            for c in range(NUM_CLASSES):
                p, t = (pred == c), (target == c)
                inter[c] += (p & t).sum().item()
                union[c] += (p | t).sum().item()

    present_classes = union > 0
    iou = inter[present_classes] / union[present_classes]   # SAME fixed formula as full-baseline run
    miou = iou.mean().item()
    print(f"per-class IoU (present): {iou}")
    print(f"classes present: {present_classes.sum().item()} / {NUM_CLASSES}")
    print(f"Dynamic INT8 mIoU (FULL): {miou:.4f}")

    # ---- latency: single-thread, matches your laptop protocol exactly, for cross-machine comparability ----
    torch.set_num_threads(1)
    dummy_s1 = torch.randn(1, 2, 96, 96)
    dummy_s2 = torch.randn(1, 12, 96, 96)
    with torch.no_grad():
        for _ in range(5):
            croma_q(SAR_images=dummy_s1, optical_images=dummy_s2)
        t0 = time.perf_counter()
        for _ in range(50):
            croma_q(SAR_images=dummy_s1, optical_images=dummy_s2)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / 50 * 1000
    print(f"Dynamic INT8 latency: {latency_ms:.2f} ms/image")

    with open("results_dynamic_quant_full.txt", "w") as f:
        f.write(f"{miou:.4f},{latency_ms:.2f},{size_mb:.2f}")

if __name__ == "__main__":
    main()