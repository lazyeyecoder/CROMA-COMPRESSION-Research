import torch, time, random
from torch.utils.data import Dataset, DataLoader
from use_croma import PretrainedCROMA

device = "cpu"

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
        return img[:12], img[12:], self.labels[i].long()   # s2, s1, label

def extract(croma, split, n):
    dl = DataLoader(DFCDataset(split, n=n), batch_size=16, num_workers=0)
    feats, labels = [], []
    with torch.no_grad():
        for s2, s1, lbl in dl:
            out = croma(SAR_images=s1, optical_images=s2)
            feats.append(out["joint_encodings"])
            labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)

def main():
    croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96).to(device)
    croma.eval()
    for p in croma.parameters():
        p.requires_grad = False

    print("extracting FULL train set (46,152 images)...")
    t0 = time.perf_counter()
    train_feats, train_lbls = extract(croma, "train", None)
    print(f"train done in {(time.perf_counter()-t0)/60:.1f} min")

    print("extracting FULL val set (8,874 images)...")
    t0 = time.perf_counter()
    val_feats, val_lbls = extract(croma, "validation", None)
    print(f"val done in {(time.perf_counter()-t0)/60:.1f} min")

    torch.save({"feats": train_feats, "lbls": train_lbls}, "dfc_train_features_full.pt")
    torch.save({"feats": val_feats, "lbls": val_lbls}, "dfc_val_features_full.pt")
    print("done — full dataset features cached")

if __name__ == "__main__":
    main()