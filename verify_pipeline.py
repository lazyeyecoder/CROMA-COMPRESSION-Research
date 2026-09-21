import os
import glob
import time
import torch
import numpy as np
import rasterio
from rasterio.enums import Resampling
from use_croma import PretrainedCROMA

device = "cpu"

# ==========================================
# 1. Official CROMA Normalization
# ==========================================
def normalize(x, use_8_bit=False):
    x = x.float()
    imgs = []
    for channel in range(x.shape[1]):
        mean_val = x[:, channel, :, :].mean()
        std_val = x[:, channel, :, :].std()
        min_value = mean_val - 2 * std_val
        max_value = mean_val + 2 * std_val

        if use_8_bit:
            img = (x[:, channel, :, :] - min_value) / (max_value - min_value + 1e-8) * 255.0
            img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)
        else:
            img = (x[:, channel, :, :] - min_value) / (max_value - min_value + 1e-8)
            img = torch.clip(img, 0, 1).unsqueeze(dim=1)
        imgs.append(img)

    return torch.cat(imgs, dim=1)

# ==========================================
# 2. Helper to Load a Batch from benv1_14k
# ==========================================
def read_and_resize(filepath, target_shape=(120, 120)):
    """Reads a tif file and forces it to a target shape (120x120)."""
    with rasterio.open(filepath) as dataset:
        data = dataset.read(
            1,
            out_shape=target_shape,
            resampling=Resampling.bilinear
        )
    return data

def load_ben14k_batch(dataset_dir="benv1_14k", batch_size=4):
    """
    Loads a batch of spatially aligned S1 (SAR) and S2 (Optical) patches 
    from the local benv1_14k directory.
    """
    s1_dirs = sorted(glob.glob(os.path.join(dataset_dir, "s1", "*")))
    s2_dirs = sorted(glob.glob(os.path.join(dataset_dir, "s2", "*")))

    if not s1_dirs or not s2_dirs:
        raise ValueError(f"Could not find subdirectories inside {dataset_dir}/s1 or {dataset_dir}/s2. Check your paths!")

    sar_tensors = []
    opt_tensors = []

    for i in range(min(batch_size, len(s1_dirs))):
        s1_path = s1_dirs[i]
        s2_path = s2_dirs[i]

        # Load S1 (2 Channels)
        if os.path.isfile(s1_path) and s1_path.endswith('.npy'):
            s1_arr = np.load(s1_path)
        else:
            tif_files = sorted(glob.glob(os.path.join(s1_path, "*.tif")))
            if len(tif_files) < 2:
                raise FileNotFoundError(f"Not enough .tif files in {s1_path}")
            # Resize all SAR bands to 120x120
            bands = [read_and_resize(f) for f in tif_files[:2]]
            s1_arr = np.stack(bands, axis=0)

        # Load S2 (12 Channels)
        if os.path.isfile(s2_path) and s2_path.endswith('.npy'):
            s2_arr = np.load(s2_path)
        else:
            tif_files = sorted(glob.glob(os.path.join(s2_path, "*.tif")))
            # Drop B10 and resize all optical bands to 120x120
            bands = [read_and_resize(f) for f in tif_files if "B10" not in f][:12]
            if len(bands) != 12:
                raise ValueError(f"Expected 12 bands in {s2_path}, found {len(bands)}")
            s2_arr = np.stack(bands, axis=0)

        sar_tensors.append(torch.from_numpy(s1_arr).float())
        opt_tensors.append(torch.from_numpy(s2_arr).float())

    sar_batch = torch.stack(sar_tensors)
    opt_batch = torch.stack(opt_tensors)

    return sar_batch, opt_batch


# ==========================================
# 3. Execution Pipeline
# ==========================================
if __name__ == "__main__":
    print("1. Loading benv1_14k sample batch...")
    try:
        sar_raw, opt_raw = load_ben14k_batch("benv1_14k", batch_size=4)
        print(f"   Raw SAR shape:     {sar_raw.shape}")
        print(f"   Raw Optical shape: {opt_raw.shape}\n")
    except Exception as e:
        print(f"Error loading files directly from folder: {e}")
        print("Exiting pipeline. Please check the error above.")
        exit(1)

    print("2. Applying Official CROMA Normalization...")
    sar_batch = normalize(sar_raw, use_8_bit=False).to(device)
    opt_batch = normalize(opt_raw, use_8_bit=False).to(device)
    print("   Normalization complete!\n")

    print("3. Loading CROMA Baseline Model...")
    model = PretrainedCROMA(
        pretrained_path="CROMA_base.pt",
        size="base",
        modality="both",
        image_resolution=120
    )
    model.to(device)
    model.eval()
    print("   Model loaded successfully!\n")

    print("4. Running Forward Pass & Measuring Latency...")
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(SAR_images=sar_batch, optical_images=opt_batch)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    print("\n--- BASELINE SUCCESS ---")
    print(f"Joint Feature Output Shape ('joint_GAP'):      {outputs['joint_GAP'].shape}")
    print(f"Joint Patch Encoding Shape ('joint_encodings'): {outputs['joint_encodings'].shape}")
    print(f"Inference Latency for 4 images:                {latency_ms:.2f} ms")
    print(f"Average Latency per image:                    {latency_ms / 4:.2f} ms")