import torch

file_path = "best_ben14k_isro_retrieval.pt"

print("Loading dataset...")
# weights_only=False allows PyTorch to load datasets containing NumPy structures
data = torch.load(file_path, map_location="cpu", weights_only=False)

print("\n--- DATASET STRUCTURE ---")
if isinstance(data, dict):
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"Key: '{key}' | Tensor Shape: {value.shape}")
        elif hasattr(value, '__len__'):
            print(f"Key: '{key}' | Type: {type(value)} | Length: {len(value)}")
        else:
            print(f"Key: '{key}' | Type: {type(value)}")
elif isinstance(data, list):
    print(f"Dataset is a List containing {len(data)} items.")
    print(f"Structure of first item: {type(data[0])}")
else:
    print(f"Dataset loaded with type: {type(data)}")