#downloading croma thing not marida
import os
import urllib.request

files_to_download = {
    "use_croma.py": "https://raw.githubusercontent.com/antofuller/CROMA/main/use_croma.py",
    "CROMA_base.pt": "https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt"
}

headers = {'User-Agent': 'Mozilla/5.0'}

for filename, url in files_to_download.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully saved {filename}!\n")
    else:
        print(f"'{filename}' already exists locally.\n")

print("All missing files are downloaded and ready!")


#marida download
# import os
# import urllib.request
# import torch

# file_name = "MARIDA_preprocessed.pt"
# url = f"https://huggingface.co/datasets/antofuller/CROMA_benchmarks/resolve/main/{file_name}"

# # 1. Download MARIDA (~577 MB)
# if not os.path.exists(file_name):
#     print("Downloading lightweight MARIDA dataset (~577 MB)...")
#     urllib.request.urlretrieve(url, file_name)
#     print("Download Complete!\n")
# else:
#     print(f"'{file_name}' already exists locally!\n")

# # 2. Load and inspect tensors safely
# data = torch.load(file_name, map_location="cpu", weights_only=False)

# print("--- MARIDA DATASET STRUCTURE ---")
# if isinstance(data, dict):
#     for key, value in data.items():
#         if hasattr(value, 'shape'):
#             print(f"Key: '{key}' | Tensor Shape: {value.shape}")
#         elif hasattr(value, '__len__'):
#             print(f"Key: '{key}' | Type: {type(value)} | Length: {len(value)}")
#         else:
#             print(f"Key: '{key}' | Type: {type(value)}")