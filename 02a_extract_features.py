import torch, time, random
from torch.utils.data import Dataset, DataLoader
from use_croma import PretrainedCROMA

device = "cpu"
PATCH = 8
N_TRAIN = 2000
N_VAL   = 1000

class DFCDataset(Dataset):
    def __init__(self, split, n=None, seed=0):
        d = torch.load("./data/DFC_preprocessed.pt")
        imgs = d[f"{split}_images"]
        lbls = d[f"{split}_labels"]
        if n is not None and n < len(imgs):
            random.seed(seed)
            idx = random.sample(range(len(imgs)), n)
            imgs, lbls = imgs[idx], lbls[idx]
        self.images, self.labels = imgs, lbls

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        img = self.images[i].float() / 255.0
        return img[:12], img[12:], self.labels[i].long()

def extract(croma, split, n):
    dl = DataLoader(DFCDataset(split, n=n), batch_size=16, num_workers=0)  # <-- fix
    feats, labels = [], []
    with torch.no_grad():
        for s2, s1, lbl in dl:
            out = croma(SAR_images=s1, optical_images=s2)
            feats.append(out["joint_encodings"])
            labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)

def main():
    tiny_dl = DataLoader(DFCDataset("train", n=32), batch_size=8, num_workers=0)  # <-- fix
    croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96).to(device)
    croma.eval()
    for p in croma.parameters():
        p.requires_grad = False

    t0 = time.perf_counter()
    with torch.no_grad():
        for s2, s1, lbl in tiny_dl:
            croma(SAR_images=s1, optical_images=s2)
    t1 = time.perf_counter()
    per_img = (t1 - t0) / 32
    print(f"measured: {per_img*1000:.0f} ms/image on this machine")
    print(f"estimated full extraction time for {N_TRAIN+N_VAL} images: {(per_img*(N_TRAIN+N_VAL))/60:.1f} min")

    print("extracting train features...")
    train_feats, train_lbls = extract(croma, "train", N_TRAIN)
    print("extracting val features...")
    val_feats, val_lbls = extract(croma, "validation", N_VAL)

    torch.save({"feats": train_feats, "lbls": train_lbls}, "dfc_train_features.pt")
    torch.save({"feats": val_feats, "lbls": val_lbls}, "dfc_val_features.pt")
    print("done — features cached")

if __name__ == "__main__":   # <-- THE actual fix, required on Windows for any DataLoader use
    main()