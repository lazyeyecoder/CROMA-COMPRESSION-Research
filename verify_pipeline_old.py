import time
import torch
from use_croma import PretrainedCROMA

device = "cpu"

print("1. Loading MARIDA dataset...")
dataset = torch.load("MARIDA_preprocessed.pt", map_location=device, weights_only=False)

# Inspect tensor shapes
print(f"   Radar tensor shape:   {dataset['radar'].shape}")
print(f"   Optical tensor shape: {dataset['optical'].shape}\n")

print("2. Loading CROMA Baseline Model...")
model = PretrainedCROMA(
    pretrained_path="CROMA_base.pt",
    size="base",
    modality="both",
    image_resolution=120
)
model.to(device)
model.eval()
print("   Model loaded successfully!\n")

# 3. Grab a test batch of 4 satellite images
sar_batch = dataset['radar'][:4].float().to(device)
opt_batch = dataset['optical'][:4].float().to(device)

print("3. Running Forward Pass & Measuring Baseline Latency...")
start_time = time.perf_counter()

with torch.no_grad():
    # Pass SAR + Optical images through CROMA to extract embeddings
    joint_embeddings, optical_embeddings, sar_embeddings = model(
        SAR_images=sar_batch, 
        optical_images=opt_batch
    )

end_time = time.perf_counter()
latency_ms = (end_time - start_time) * 1000

print("--- BASELINE SUCCESS ---")
print(f"Joint Feature Output Shape: {joint_embeddings.shape}")
print(f"Inference Latency for 4 images: {latency_ms:.2f} ms")
print(f"Average Latency per image:     {latency_ms / 4:.2f} ms")