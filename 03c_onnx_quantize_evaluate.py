import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from use_croma import PretrainedCROMA
import random
from tqdm import tqdm
import onnx
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.quant_utils import QuantizationMode
import onnxruntime as ort
import numpy as np

NUM_CLASSES = 8
PATCH = 8

class DFCDataset(Dataset):
    def __init__(self, split="validation", n=None, seed=0):
        d = torch.load("./data/DFC_preprocessed.pt")
        imgs = d[f"{split}_images"]
        lbls = d[f"{split}_labels"]
        if n is not None and n < len(imgs):
            random.seed(seed)
            idx = random.sample(range(len(imgs)), n)
            imgs = [imgs[i] for i in idx] if isinstance(imgs, list) else imgs[idx]
            lbls = [lbls[i] for i in idx] if isinstance(lbls, list) else lbls[idx]

        self.images, self.labels = imgs, lbls
        
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = self.images[i].float() / 255.0
        return img[12:], img[:12], self.labels[i].long()

def downsample_labels(labels, patch=PATCH):
    B, H, W = labels.shape
    ph, pw = H // patch, W // patch
    x = (
        labels.view(B, ph, patch, pw, patch)
        .permute(0, 1, 3, 2, 4)
        .reshape(B, ph * pw, patch * patch)
    )
    return torch.mode(x, dim=-1).values

print("1. Loading FP32 Baseline CROMA...")
croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96).to("cpu")
croma.eval()

print("2. Exporting Baseline to ONNX...")
dummy_s1 = torch.randn(1, 2, 96, 96)
dummy_s2 = torch.randn(1, 12, 96, 96)

torch.onnx.export(
    croma, 
    (dummy_s1, dummy_s2), 
    "CROMA_baseline.onnx", 
    export_params=True,
    opset_version=18, 
    do_constant_folding=True,
    input_names=['SAR_images', 'optical_images'],
    output_names=['joint_encodings']
)

print("3. Quantizing ONNX Model to INT8 (Bypassing Shape Inference)...")
# We load the model directly
model = onnx.load("CROMA_baseline.onnx")

# We instantiate the quantizer directly, which allows us to skip the shape inference step
# that causes the crash in quantize_dynamic
quantizer = ONNXQuantizer(
    model=model,
    per_channel=False,
    reduce_range=False,
    mode=QuantizationMode.IntegerOps,
    static=False,
    weight_qType=QuantType.QInt8,
    activation_qType=QuantType.QUInt8,
    tensors_range=None,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    op_types_to_quantize=None,
    extra_options={'EnableSubgraph': True}
)

# Run the quantization
quantizer.quantize_model()

# Save it manually
quantizer.model.save_model_to_file("CROMA_INT8.onnx")

print("4. Setting up ONNX Runtime Inference...")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

ort_session = ort.InferenceSession("CROMA_INT8.onnx", sess_options, providers=['CPUExecutionProvider'])

probe = nn.Linear(768, NUM_CLASSES).to("cpu")
probe.load_state_dict(torch.load("probe_baseline.pt"))
probe.eval()

val_ds = DFCDataset("validation", n=1000)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

print("5. Evaluating ONNX INT8 Quantized mIoU...")
inter = torch.zeros(NUM_CLASSES)
union = torch.zeros(NUM_CLASSES)

for s1, s2, lbl in tqdm(val_dl, desc="ONNX Inference"):
    # Must use batch size of 1 since we exported without dynamic axes
    for i in range(s1.shape[0]):
        single_s1 = s1[i:i+1].numpy()
        single_s2 = s2[i:i+1].numpy()
        
        ort_outs = ort_session.run(None, {'SAR_images': single_s1, 'optical_images': single_s2})
        feats = torch.tensor(ort_outs[0]) 
        
        with torch.no_grad():
            pred = probe(feats).argmax(-1)
            target = downsample_labels(lbl[i:i+1]).clamp(min=0)
            for c in range(NUM_CLASSES):
                p_mask, t_mask = (pred == c), (target == c)
                inter[c] += (p_mask & t_mask).sum().item()
                union[c] += (p_mask | t_mask).sum().item()

miou = (inter / union.clamp(min=1)).mean().item()

print("6. Measuring ONNX Latency (Single Thread)...")
for _ in range(5):
    ort_session.run(None, {'SAR_images': dummy_s1.numpy(), 'optical_images': dummy_s2.numpy()})

t0 = time.perf_counter()
for _ in range(50):
    ort_session.run(None, {'SAR_images': dummy_s1.numpy(), 'optical_images': dummy_s2.numpy()})
t1 = time.perf_counter()

latency_ms = ((t1 - t0) / 50) * 1000
model_size_mb = os.path.getsize("CROMA_INT8.onnx") / (1024 * 1024)

print("\n--- ONNX INT8 QUANTIZED RESULTS ---")
print(f"1. Quantized mIoU:      {miou:.4f}")
print(f"2. Single-CPU Latency:  {latency_ms:.2f} ms/image")
print(f"3. Model File Size:     {model_size_mb:.2f} MB")