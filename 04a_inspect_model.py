import torch
from use_croma import PretrainedCROMA

croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96)

print("=== s2_encoder structure ===")
print(croma.s2_encoder)
print("\n=== one layer's named parameters (s2_encoder) ===")
for name, p in croma.s2_encoder.named_parameters():
    print(name, tuple(p.shape))
    if "0." in name and name.count(".") <= 3:  # just first layer, keep it short
        continue

print("\n=== joint_encoder structure (just to see the shape, not pruning it yet) ===")
print(croma.joint_encoder)