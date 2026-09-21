from huggingface_hub import hf_hub_download
for f in ["DFC_preprocessed.pt", "Canadian_Cropland_preprocessed.pt"]:
    hf_hub_download(repo_id="antofuller/CROMA_benchmarks", filename=f, repo_type="dataset", local_dir="./data")
print("done")
import torch, time
from use_croma import PretrainedCROMA

croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96)
croma.eval()
dummy_s1, dummy_s2 = torch.randn(8,2,96,96), torch.randn(8,12,96,96)  # batch of 8
with torch.no_grad():
    for _ in range(3): croma(SAR_images=dummy_s1, optical_images=dummy_s2)  # warmup
    t0 = time.perf_counter()
    for _ in range(10): croma(SAR_images=dummy_s1, optical_images=dummy_s2)
    t1 = time.perf_counter()
per_img_ms = (t1-t0)/10/8*1000
print(f"speed on this PC: {per_img_ms:.1f} ms/image")
print(f"full DFC2020 (55,026 imgs) extraction estimate: {per_img_ms*55026/60000:.1f} min")