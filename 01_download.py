# import os
# from huggingface_hub import hf_hub_download

# print("Downloading official preprocessed DFC2020 benchmark dataset...")
# os.makedirs("./data", exist_ok=True)

# hf_hub_download(
#     repo_id="antofuller/CROMA_benchmarks",
#     filename="DFC_preprocessed.pt",
#     repo_type="dataset",
#     local_dir="./data",
# )
# print("SUCCESS: DFC_preprocessed.pt is downloaded and ready in ./data!")

from huggingface_hub import hf_hub_download
for f in ["DFC_preprocessed.pt", "Canadian_Cropland_preprocessed.pt"]:
    hf_hub_download(repo_id="antofuller/CROMA_benchmarks", filename=f, repo_type="dataset", local_dir="./data")
print("done")