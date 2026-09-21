import os, time, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.quantization as quant
from use_croma import PretrainedCROMA
from tqdm import tqdm

torch.backends.quantized.engine = 'onednn'   # fbgemm not available on this build

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
        return img[:12], img[12:], self.labels[i].long()  # s2, s1, lbl

def downsample_labels(labels, patch=PATCH):
    B, H, W = labels.shape
    ph, pw = H // patch, W // patch
    x = labels.view(B, ph, patch, pw, patch).permute(0, 1, 3, 2, 4).reshape(B, ph*pw, patch*patch)
    return torch.mode(x, dim=-1).values.clamp(min=0)

def main():
    # ---- 1. load FP32, prep for static quant ----
    croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96).to(device)
    croma.eval()
    croma.qconfig = quant.get_default_qconfig('onednn')
    quant.prepare(croma, inplace=True)

    # ---- 2. calibrate (THIS is the step the eval script was skipping) ----
    calib_dl = DataLoader(DFCDataset("train", n=32), batch_size=8, num_workers=0)
    print("calibrating...")
    with torch.no_grad():
        for s2, s1, _ in tqdm(calib_dl, desc="Calibrating"):
            croma(SAR_images=s1, optical_images=s2)

    # ---- 3. convert (now observers actually have real min/max stats) ----
    quant.convert(croma, inplace=True)
    torch.save(croma.state_dict(), "CROMA_INT8_Static.pt")  # keep for record, don't reload it
    size_mb = os.path.getsize("CROMA_INT8_Static.pt") / (1024*1024)
    print(f"quantized, saved, size = {size_mb:.2f} MB")

    # ---- 4. evaluate mIoU — SAME croma object, still in memory, still quantized ----
    val_dl = DataLoader(DFCDataset("validation", n=1000), batch_size=16, num_workers=0)
    probe = nn.Linear(768, NUM_CLASSES).to(device)
    probe.load_state_dict(torch.load("probe_baseline.pt"))
    probe.eval()

    inter = torch.zeros(NUM_CLASSES); union = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for s2, s1, lbl in tqdm(val_dl, desc="Eval mIoU"):
            out = croma(SAR_images=s1, optical_images=s2)
            pred = probe(out["joint_encodings"]).argmax(-1)
            target = downsample_labels(lbl)
            for c in range(NUM_CLASSES):
                p, t = (pred == c), (target == c)
                inter[c] += (p & t).sum().item()
                union[c] += (p | t).sum().item()
    miou = (inter / union.clamp(min=1)).mean().item()
    print(f"Static INT8 mIoU: {miou:.4f}")

    # ---- 5. latency — SAME thread-pin, SAME warmup style, so it's comparable to whatever FP32 number you re-measure below ----
    torch.set_num_threads(1)
    dummy_s1 = torch.randn(1, 2, 96, 96)
    dummy_s2 = torch.randn(1, 12, 96, 96)
    with torch.no_grad():
        for _ in range(5):
            croma(SAR_images=dummy_s1, optical_images=dummy_s2)
        t0 = time.perf_counter()
        for _ in range(50):
            croma(SAR_images=dummy_s1, optical_images=dummy_s2)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / 50 * 1000
    print(f"Static INT8 latency: {latency_ms:.2f} ms/image")

    with open("results_static_quant.txt", "w") as f:
        f.write(f"{miou:.4f},{latency_ms:.2f},{size_mb:.2f}")

if __name__ == "__main__":
    main()