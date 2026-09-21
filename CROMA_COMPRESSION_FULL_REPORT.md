# CROMA Multimodal Vision Transformer Compression

## Full Codebase and Experimental Report

**Workspace:** `Research Honours/code`  
**Report scope:** CROMA architecture, preprocessing, downstream segmentation probe, quantization, soft pruning, structural pruning, hybrid compression, ONNX export, CPU benchmarking, recorded results, and reproducibility caveats.  
**Primary task:** DFC2020 patch-level land-cover classification/segmentation using Sentinel-1 SAR and Sentinel-2 optical imagery.

---

## 1. Executive Summary

The primary experimental pipeline uses the pretrained CROMA base model implemented in [`use_croma.py`](./use_croma.py). It processes:

- Sentinel-1 SAR imagery with 2 channels.
- Sentinel-2 optical imagery with 12 channels.
- 96 × 96 pixel tiles in the DFC2020 compression experiments.
- 8 × 8 non-overlapping patches, producing 144 tokens.
- A 768-dimensional embedding.
- 16 attention heads with 48 dimensions per head.
- 12 optical transformer layers.
- 6 SAR transformer layers.
- 6 joint self-/cross-attention layers.
- A 2D Euclidean-distance ALiBi-style relative spatial bias.
- An 8-class linear segmentation probe.

The main compression families are:

1. Dynamic PyTorch INT8 quantization of `nn.Linear` modules.
2. Soft attention-head pruning by zeroing low-L1-norm Q/K/V rows.
3. Structural attention-head pruning by rebuilding smaller attention matrices.
4. Hybrid pruning followed by dynamic INT8 quantization.
5. Separate static PyTorch INT8 and ONNX INT8 prototypes.

The largest recorded storage reduction is obtained by dynamic INT8 quantization:

```text
FP32 checkpoint: approximately 741.54 MB
Dynamic INT8 checkpoint: approximately 186.20 MB
```

The structural-plus-dynamic-INT8 artifact is approximately 173.54 MB.

The workspace contains multiple script generations, subset evaluations, and overwritten result files. Consequently, some recorded values cannot be treated as independently reproducible until the corresponding scripts are rerun and their output files are regenerated.

---

## 2. Workspace Inventory

### 2.1 Core model and checkpoints

| File | Purpose |
|---|---|
| [`use_croma.py`](./use_croma.py) | Primary CROMA implementation |
| `CROMA_base.pt` | Pretrained FP32 base checkpoint |
| `CROMA_baseline.onnx` | ONNX export of the actual CROMA pipeline |
| `CROMA_baseline.onnx.data` | External ONNX tensor data |
| `CROMA_baseline_cleared.onnx` | Additional ONNX artifact |
| `CROMA_baseline_preprocessed.onnx` | Additional ONNX artifact |
| [`pruned_arch.py`](./pruned_arch.py) | Reconstructs the structurally pruned CROMA architecture |
| `probe_baseline.pt` | Trained 768-to-8 linear probe |

### 2.2 Evaluation and training scripts

| File | Purpose |
|---|---|
| [`02_baseline_eval.py`](./02_baseline_eval.py) | FP32 DFC2020 evaluation |
| [`02a_extract_features.py`](./02a_extract_features.py) | Feature extraction |
| [`02b_train_probe.py`](./02b_train_probe.py) | Linear probe training |
| [`00_fp32_latency.py`](./00_fp32_latency.py) | Single-thread FP32 latency measurement |
| [`06_generate_results.py`](./06_generate_results.py) | Combines result files into a summary table |

### 2.3 Quantization scripts

| File | Purpose |
|---|---|
| [`03_quantize_dynamic.py`](./03_quantize_dynamic.py) | Dynamic PyTorch INT8 quantization |
| [`03e_dynamic_quant_clean.py`](./03e_dynamic_quant_clean.py) | Clean dynamic INT8 evaluation on a 1,000-image subset |
| [`03a_quantize_subset_static.py`](./03a_quantize_subset_static.py) | Static PyTorch INT8 preparation/calibration/conversion |
| [`03b_evaluate_static.py`](./03b_evaluate_static.py) | Static INT8 state-dict reconstruction/evaluation |
| [`03d_quantize_and_eval.py`](./03d_quantize_and_eval.py) | Calibrated static INT8 evaluation using oneDNN |
| [`03c_onnx_quantize_evaluate.py`](./03c_onnx_quantize_evaluate.py) | Actual CROMA ONNX export and direct ONNX quantization |
| [`00b_check_quant_engines.py`](./00b_check_quant_engines.py) | Checks available PyTorch quantized engines |

### 2.4 Pruning and hybrid scripts

| File | Purpose |
|---|---|
| [`04_prune.py`](./04_prune.py) | Earlier 20% soft head-pruning implementation |
| [`04_prune_subset.py`](./04_prune_subset.py) | 20% soft pruning with Q/K/V scoring and a 1,000-image subset |
| [`04b_structural_prune.py`](./04b_structural_prune.py) | Physical 13-of-16 head removal |
| [`05_hybrid.py`](./05_hybrid.py) | Soft-pruned model followed by dynamic INT8 |
| [`05_hybrid_subset.py`](./05_hybrid_subset.py) | Subset version of soft-pruned plus INT8 hybrid |
| [`05a_hybrid_structural.py`](./05a_hybrid_structural.py) | Structurally pruned model followed by dynamic INT8 |
| [`00_notebook_onnx.py`](./00_notebook_onnx.py) | Separate representative ONNX pipeline with 25% head zeroing |

### 2.5 Recorded output files

| File | Stored format |
|---|---|
| [`results_fp32_latency.txt`](./results_fp32_latency.txt) | `latency_ms` only |
| [`results_dynamic_quant.txt`](./results_dynamic_quant.txt) | `mIoU,latency_ms,size_MB` |
| [`results_quant.txt`](./results_quant.txt) | `mIoU,latency_ms,size_MB` |
| [`results_pruned20.txt`](./results_pruned20.txt) | `mIoU,latency_ms,size_MB` |
| [`results_structural_pruned.txt`](./results_structural_pruned.txt) | `mIoU,latency_ms,size_MB` |
| [`results_hybrid.txt`](./results_hybrid.txt) | `mIoU,latency_ms,size_MB` |
| [`results_hybrid_structural.txt`](./results_hybrid_structural.txt) | `mIoU,latency_ms,size_MB` |

The workspace also contains a historical verification log in [`Code history for verify_pipeline.txt`](./Code%20history%20for%20verify_pipeline.txt).

---

## 3. Primary CROMA Model Architecture

## 3.1 Model entry point

The primary model class is:

```python
class PretrainedCROMA(nn.Module):
```

in [`use_croma.py`](./use_croma.py).

The constructor accepts:

```python
PretrainedCROMA(
    pretrained_path="CROMA_base.pt",
    size="base",
    modality="both",
    image_resolution=96,
)
```

The constructor requires the image resolution to be divisible by 8.

## 3.2 Input modalities and tensor shapes

The model has fixed channel definitions:

```python
self.s1_channels = 2
self.s2_channels = 12
```

The project dataset stores a 14-channel tensor:

```text
Dataset tensor: [N, 14, 96, 96]
```

The scripts split this tensor as follows:

```python
s2 = img[:12]   # Sentinel-2 optical
s1 = img[12:]   # Sentinel-1 SAR
```

Therefore, for a batch:

```text
SAR input:     [B, 2, 96, 96]
Optical input: [B, 12, 96, 96]
```

Inputs are converted from stored image values to floating point and normalized by:

```python
img = self.images[i].float() / 255.0
```

This preprocessing appears in the baseline and compression evaluation scripts.

## 3.3 Base and large configurations

The model configuration in [`use_croma.py`](./use_croma.py) is:

| Configuration | Hidden dimension | Encoder depth | Heads | Patch size |
|---|---:|---:|---:|---:|
| Base | 768 | 12 | 16 | 8 |
| Large | 1024 | 24 | 16 | 8 |

All compression experiments instantiate `size="base"`.

## 3.4 Patchification

The ViT implementation uses:

```python
self.patch_size = 8
```

Each image is rearranged into non-overlapping patches:

```python
x = rearrange(
    imgs,
    "b c (h i) (w j) -> b (h w) (c i j)",
    i=self.patch_size,
    j=self.patch_size,
)
```

For 96 × 96 inputs:

```text
Patches per dimension: 96 / 8 = 12
Total patches:         12 × 12 = 144
```

Patch input dimensions:

```text
SAR patch:     2 × 8 × 8  = 128
Optical patch: 12 × 8 × 8 = 768
```

Each flattened patch is projected to 768 dimensions by a linear layer.

## 3.5 Transformer depth

The base model creates:

### Optical branch

```python
self.s2_encoder = ViT(
    dim=768,
    depth=12,
    in_channels=12,
)
```

The optical branch contains 12 self-attention/FFN blocks.

### SAR branch

```python
self.s1_encoder = ViT(
    dim=768,
    depth=int(12 / 2),
    in_channels=2,
)
```

The SAR branch contains 6 self-attention/FFN blocks.

### Joint branch

```python
self.cross_encoder = BaseTransformerCrossAttn(
    dim=768,
    depth=int(12 / 2),
    num_heads=16,
)
```

The joint branch contains 6 blocks. Each joint block includes:

1. Self-attention on the SAR-side sequence.
2. Cross-attention using optical features as context.
3. A feed-forward network.

## 3.6 Attention dimensions

The base hidden dimension is 768 and the number of heads is 16:

```text
Head dimension = 768 / 16 = 48
```

The standard self-attention QKV projection has shape:

```text
[3 × 16 × 48, 768] = [2304, 768]
```

The standard output projection has shape:

```text
[768, 768]
```

The cross-attention module has separate Q, K, and V projections, each operating at 768-dimensional full width before head reshaping.

## 3.7 Output features

For the 96 × 96 base experiment:

```text
SAR encodings:      [B, 144, 768]
Optical encodings:  [B, 144, 768]
Joint encodings:    [B, 144, 768]
SAR GAP:            [B, 768]
Optical GAP:        [B, 768]
Joint GAP:          [B, 768]
```

The segmentation pipeline uses:

```python
feats = out["joint_encodings"]
```

not `joint_GAP`.

---

## 4. 2D-ALiBi Spatial Encoding

## 4.1 Primary implementation

The primary CROMA model uses the `get_2dalibi()` function in [`use_croma.py`](./use_croma.py).

The spatial coordinates are generated over a square patch grid:

```python
points = list(
    itertools.product(
        range(int(math.sqrt(num_patches))),
        range(int(math.sqrt(num_patches))),
    )
)
```

For each query/key patch pair, the implementation computes Euclidean distance:

```text
d(i, j) = sqrt((row_i - row_j)^2 + (col_i - col_j)^2)
```

Each attention head has a slope. The resulting bias is negative:

```text
bias[h, i, j] = -slope[h] × d(i, j)
```

For 96 × 96 inputs:

```text
num_patches = 144
ALiBi shape = [1, 16, 144, 144]
```

The bias is added to the attention logits before softmax for both self-attention and cross-attention.

## 4.2 Representative ONNX implementation

The separate [`00_notebook_onnx.py`](./00_notebook_onnx.py) defines:

```python
class Spatial2DALiBi(nn.Module):
```

It uses a fixed default grid:

```python
grid_size=(15, 15)
```

and computes:

```python
coords = torch.stack(
    torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
        indexing="ij",
    ),
    dim=-1,
).float()

coords = coords.view(-1, 2)
dist = torch.cdist(coords, coords, p=2)
bias = -dist.unsqueeze(0) * self.slopes
```

This prototype therefore uses:

```text
15 × 15 = 225 tokens
```

It should not be confused with the 96 × 96 primary CROMA experiment, which uses 144 tokens.

---

## 5. Downstream Segmentation Probe

The probe is trained in [`02b_train_probe.py`](./02b_train_probe.py) and loaded by the compression scripts.

Configuration:

```python
NUM_CLASSES = 8
probe = nn.Linear(768, NUM_CLASSES)
```

Therefore:

```text
Probe input:  [B, 144, 768]
Probe output: [B, 144, 8]
```

The training loss is:

```python
loss = F.cross_entropy(
    logits.reshape(-1, NUM_CLASSES),
    targets.reshape(-1),
)
```

The probe is trained for five epochs using AdamW with learning rate `1e-3`.

The CROMA backbone is frozen during probe training:

```python
for p in croma.parameters():
    p.requires_grad = False
```

## 5.1 Label downsampling

The label downsampling function is duplicated across the evaluation scripts. Its logic is:

```python
def downsample_labels(labels, patch=8):
    B, H, W = labels.shape
    ph, pw = H // patch, W // patch

    x = labels.view(
        B,
        ph,
        patch,
        pw,
        patch,
    )

    x = x.permute(
        0,
        1,
        3,
        2,
        4,
    )

    x = x.reshape(
        B,
        ph * pw,
        patch * patch,
    )

    return torch.mode(x, dim=-1).values
```

For 96 × 96 labels and patch size 8:

```text
Original labels: [B, 96, 96]
Patch grid:      12 × 12
Pixels per patch: 64
Downsampled:     [B, 144]
```

Each 8 × 8 image patch receives the modal class among its 64 pixel labels.

Several evaluation scripts apply:

```python
target = downsample_labels(lbl).clamp(min=0)
```

---

## 6. Dynamic INT8 Quantization

## 6.1 PyTorch dynamic quantization

The main implementation is in [`03_quantize_dynamic.py`](./03_quantize_dynamic.py):

```python
croma_int8 = torch.quantization.quantize_dynamic(
    croma,
    {nn.Linear},
    dtype=torch.qint8,
)
```

Technical specification:

| Item | Value |
|---|---|
| API | `torch.quantization.quantize_dynamic` |
| Target modules | `nn.Linear` |
| Weight dtype | `torch.qint8` |
| Activation quantization | Dynamic at inference time |
| Saved artifact | `croma_int8.pt` |
| Alternate saved artifact | `CROMA_INT8_Dynamic.pt` |

The probe is loaded separately:

```python
probe.load_state_dict(
    torch.load("probe_baseline.pt")
)
```

The quantized CROMA state dictionary is saved after quantization.

## 6.2 ONNX Runtime dynamic quantization

The representative ONNX pipeline imports:

```python
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)
```

and calls:

```python
quantize_dynamic(
    model_input=fp32_onnx_path,
    model_output=int8_onnx_path,
    weight_type=QuantType.QUInt8,
)
```

This implementation is in [`00_notebook_onnx.py`](./00_notebook_onnx.py).

## 6.3 Direct ONNX quantizer

The actual CROMA ONNX script [`03c_onnx_quantize_evaluate.py`](./03c_onnx_quantize_evaluate.py) directly constructs `ONNXQuantizer`:

```python
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
    extra_options={
        "EnableSubgraph": True,
    },
)

quantizer.quantize_model()
quantizer.model.save_model_to_file(
    "CROMA_INT8.onnx"
)
```

The script uses QInt8 weights and QUInt8 activations, with per-channel quantization disabled.

---

## 7. Static INT8 Quantization

Although the primary requested comparison focuses on dynamic INT8, the workspace also contains static INT8 implementations.

## 7.1 FBGEMM path

[`03a_quantize_subset_static.py`](./03a_quantize_subset_static.py) uses:

```python
croma.qconfig = quant.get_default_qconfig("fbgemm")
quant.prepare(croma, inplace=True)
```

It calibrates with:

```text
Training images: 32
Calibration batch size: 8
```

Then converts:

```python
quant.convert(croma, inplace=True)
```

and saves:

```text
CROMA_INT8_Static.pt
```

## 7.2 oneDNN path

[`03d_quantize_and_eval.py`](./03d_quantize_and_eval.py) specifies:

```python
torch.backends.quantized.engine = "onednn"
```

and:

```python
croma.qconfig = quant.get_default_qconfig("onednn")
quant.prepare(croma, inplace=True)
```

It calibrates on 32 training images, converts the same in-memory model, evaluates it, and writes:

```text
CROMA_INT8_Static.pt
```

The recorded file size of this artifact is approximately 189.25 MB, but a corresponding persistent `results_static_quant.txt` file is not present.

---

## 8. Soft Attention-Head Pruning

## 8.1 Earlier implementation

The earlier soft-pruning implementation is in [`04_prune.py`](./04_prune.py).

Configuration:

```python
NUM_HEADS = 16
PRUNE_RATIO = 0.2
HEAD_DIM = 48
```

Number of heads:

```text
Heads pruned = int(16 × 0.2) = 3
Heads retained = 13
```

The implementation accesses:

```python
w = attn.to_qkv.weight.data
```

with assumed shape:

```text
[2304, 768]
```

It computes a per-head score using the query slice:

```python
q_slice = w[
    h * head_dim:(h + 1) * head_dim
]

scores.append(
    q_slice.abs().sum().item()
)
```

The three lowest-scoring heads are selected and their Q, K, and V rows are zeroed:

```python
w[q_start:q_end] = 0
w[k_start:k_end] = 0
w[v_start:v_end] = 0
```

This is soft pruning because:

- Tensor shapes remain unchanged.
- The number of attention heads remains 16.
- Zeroed heads are still represented in the computation graph.

## 8.2 Q/K/V L1 implementation

The newer [`04_prune_subset.py`](./04_prune_subset.py) computes the combined Q/K/V score:

```python
score[h] =
    abs(Q_h).sum() +
    abs(K_h).sum() +
    abs(V_h).sum()
```

For self-attention:

```python
q, k, v = (
    w[sl],
    w[dim + sl.start:dim + sl.stop],
    w[2 * dim + sl.start:2 * dim + sl.stop],
)

scores.append(
    (q.abs().sum() +
     k.abs().sum() +
     v.abs().sum()).item()
)
```

It also separately applies Q/K/V L1 pruning to cross-attention:

```python
wq, wk, wv = (
    cross_attn.to_q.weight.data,
    cross_attn.to_k.weight.data,
    cross_attn.to_v.weight.data,
)
```

The target modules are:

- Optical self-attention.
- SAR self-attention.
- Joint encoder self-attention.
- Joint cross-attention.

---

## 9. Structural Attention-Head Pruning

Structural pruning is implemented in [`04b_structural_prune.py`](./04b_structural_prune.py) and [`pruned_arch.py`](./pruned_arch.py).

## 9.1 Configuration

```python
NUM_HEADS = 16
HEAD_DIM = 48
PRUNE_RATIO = 0.2
N_KEEP = 16 - int(16 * 0.2)
```

Therefore:

```text
Heads removed: 3
Heads retained: 13
Reduced attention width: 13 × 48 = 624
```

## 9.2 Global importance calculation

The implementation collects all relevant attention modules:

```text
Optical self-attention
SAR self-attention
Joint self-attention
Joint cross-attention
```

For every head, it accumulates Q/K/V L1 scores:

```python
scores[h] = (
    q.abs().sum() +
    k.abs().sum() +
    v.abs().sum()
)
```

The scores are accumulated across the whole model:

```python
total_score = torch.zeros(NUM_HEADS)

for module in self_modules:
    total_score += head_importance_self(module)

for module in cross_modules:
    total_score += head_importance_cross(module)
```

The top 13 heads are retained:

```python
kept_idx = (
    total_score.argsort(descending=True)[:N_KEEP]
    .sort()
    .values
    .tolist()
)
```

## 9.3 Self-attention matrix changes

Original self-attention dimensions:

```text
QKV weight: [3 × 16 × 48, 768] = [2304, 768]
Output weight: [768, 768]
```

Structural dimensions:

```text
QKV weight: [3 × 13 × 48, 768] = [1872, 768]
Output weight: [768, 13 × 48] = [768, 624]
```

The selected Q, K, and V row blocks are copied into the new smaller QKV layer. The selected input columns are copied into the output projection.

## 9.4 Cross-attention matrix changes

Original cross-attention Q/K/V matrices:

```text
[768, 768]
```

Structural cross-attention Q/K/V matrices:

```text
[624, 768]
```

Original output projection:

```text
[768, 768]
```

Structural output projection:

```text
[768, 624]
```

The reduced 624-dimensional attention output is projected back to 768 dimensions.

## 9.5 ALiBi slicing

Because the number of heads changes, the shared spatial bias is sliced:

```python
croma.attn_bias = croma.attn_bias[
    :,
    kept_idx,
    :,
    :,
]
```

The bias changes from:

```text
[1, 16, 144, 144]
```

to:

```text
[1, 13, 144, 144]
```

## 9.6 State dictionary and reconstruction

The structurally pruned model is saved as:

```text
croma_structural_pruned.pt
```

The hybrid reconstruction path calls:

```python
croma = build_structural_pruned_croma()
```

and then loads:

```python
state = torch.load(
    "croma_structural_pruned.pt"
)
croma.load_state_dict(state)
```

The architecture must be rebuilt before loading because the state dictionary contains reduced matrix dimensions.

---

## 10. Hybrid Compression

## 10.1 Soft pruning followed by dynamic INT8

Implemented in [`05_hybrid.py`](./05_hybrid.py).

Operation order:

```text
1. Instantiate baseline CROMA.
2. Load croma_pruned_20.pt.
3. Set evaluation mode.
4. Quantize all nn.Linear modules dynamically to qint8.
5. Save croma_hybrid.pt.
```

Code:

```python
croma_pruned.load_state_dict(
    torch.load("croma_pruned_20.pt")
)

croma_hybrid = torch.quantization.quantize_dynamic(
    croma_pruned,
    {nn.Linear},
    dtype=torch.qint8,
)

torch.save(
    croma_hybrid.state_dict(),
    "croma_hybrid.pt",
)
```

## 10.2 Structural pruning followed by dynamic INT8

Implemented in [`05a_hybrid_structural.py`](./05a_hybrid_structural.py).

Operation order:

```text
1. Rebuild the 13-head structural architecture.
2. Load croma_structural_pruned.pt.
3. Quantize nn.Linear modules dynamically.
4. Save croma_hybrid_structural.pt.
```

Code:

```python
croma_structural = (
    build_structural_pruned_croma()
)

croma_hybrid = torch.quantization.quantize_dynamic(
    croma_structural,
    {nn.Linear},
    dtype=torch.qint8,
)

torch.save(
    croma_hybrid.state_dict(),
    "croma_hybrid_structural.pt",
)
```

---

## 11. Benchmarking and Hardware Simulation

## 11.1 Primary PyTorch benchmark

The primary benchmark simulates a single-thread CPU:

```python
torch.set_num_threads(1)
```

Inputs:

```python
dummy_s1 = torch.randn(1, 2, 96, 96)
dummy_s2 = torch.randn(1, 12, 96, 96)
```

Warm-up:

```python
for _ in range(5):
    croma(
        SAR_images=dummy_s1,
        optical_images=dummy_s2,
    )
```

Timed iterations:

```python
t0 = time.perf_counter()

for _ in range(50):
    croma(
        SAR_images=dummy_s1,
        optical_images=dummy_s2,
    )

t1 = time.perf_counter()
latency_ms = (t1 - t0) / 50 * 1000
```

The benchmark reports the mean latency over 50 single-image inferences.

## 11.2 ONNX Runtime benchmark

The representative ONNX pipeline creates:

```python
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
```

The provider is:

```python
providers=["CPUExecutionProvider"]
```

Representative ONNX benchmark:

```text
Warm-up runs: 10
Timed iterations: 100
Metrics: average latency and P90 latency
```

The actual CROMA ONNX evaluation script uses:

```text
Warm-up runs: 5
Timed iterations: 50
Provider: CPUExecutionProvider
intra_op_num_threads: 1
inter_op_num_threads: 1
```

## 11.3 ONNX export versions

There are two separate export configurations:

| Script | Model | Opset |
|---|---|---:|
| [`00_notebook_onnx.py`](./00_notebook_onnx.py) | Representative 256-dimensional model | 14 |
| [`03c_onnx_quantize_evaluate.py`](./03c_onnx_quantize_evaluate.py) | Actual pretrained CROMA | 18 |

The representative script exports:

```python
torch.onnx.export(
    model,
    (dummy_sar, dummy_opt),
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["sar_input", "optical_input"],
    output_names=["logits"],
    dynamo=False,
)
```

The actual CROMA ONNX script exports:

```python
torch.onnx.export(
    croma,
    (dummy_s1, dummy_s2),
    "CROMA_baseline.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=[
        "SAR_images",
        "optical_images",
    ],
    output_names=["joint_encodings"],
)
```

---

## 12. Experimental Results

## 12.1 Result-file format

Most result files contain:

```text
mIoU,latency_ms,model_size_MB
```

The baseline latency file is different:

```text
latency_ms
```

## 12.2 Recorded file contents

The recorded values inspected in the workspace are:

| File | Recorded contents |
|---|---|
| `results_fp32_latency.txt` | `663.35` |
| `results_dynamic_quant.txt` | `0.4810,364.31,186.20` |
| `results_quant.txt` | `0.4863,580.47,186.19` |
| `results_pruned20.txt` | `0.2512,692.28,741.57` |
| `results_structural_pruned.txt` | `0.3767,632.92,690.95` |
| `results_hybrid.txt` | `0.2526,521.34,186.19` |
| `results_hybrid_structural.txt` | `0.3789,494.47,173.54` |

The two result files `results_dynamic_quant.txt` and `results_quant.txt` correspond to different dynamic-quantization script generations or evaluation runs:

```text
results_dynamic_quant.txt = 0.4810,364.31,186.20
results_quant.txt         = 0.4863,580.47,186.19
```

The former is associated with [`03e_dynamic_quant_clean.py`](./03e_dynamic_quant_clean.py), while the latter is associated with [`03_quantize_dynamic.py`](./03_quantize_dynamic.py).

## 12.3 Main recorded comparison

| Configuration | Model size (MB) | mIoU | Single-thread CPU latency (ms) | Primary evidence |
|---|---:|---:|---:|---|
| FP32 baseline | 741.54 | Not persisted in current workspace | 663.35 | `CROMA_base.pt`, `results_fp32_latency.txt` |
| Dynamic INT8, clean subset run | 186.20 | 0.4810 | 364.31 | `results_dynamic_quant.txt` |
| Dynamic INT8, earlier run | 186.19 | 0.4863 | 580.47 | `results_quant.txt` |
| Soft-pruned 20% | 741.57 | 0.2512 | 692.28 | `results_pruned20.txt` |
| Structural-pruned 20%, 13/16 heads retained | 690.95 | 0.3767 | 632.92 | `results_structural_pruned.txt` |
| Soft-pruned 20% + dynamic INT8 | 186.19 | 0.2526 | 521.34 | `results_hybrid.txt` |
| Structural-pruned + dynamic INT8 | 173.54 | 0.3789 | 494.47 | `results_hybrid_structural.txt` |

## 12.4 Requested 25% configuration

The workspace contains a 25% configuration only in the separate representative ONNX script:

```python
pruned_model = prune_attention_heads(
    model,
    prune_ratio=0.25,
)
```

For that model:

```text
Total heads: 8
Heads pruned: max(1, int(8 × 0.25)) = 2
Heads retained: 6
```

The representative ONNX script prints:

```text
FP32 Baseline
Standalone INT8 Quantized
Structured Pruned (25%)
Hybrid (Pruned 25% + INT8)
```

but no captured output file containing the actual size, latency, and mIoU values was found in the workspace.

The 25% representative model also differs from the primary CROMA model:

| Property | Primary CROMA | Representative ONNX model |
|---|---:|---:|
| Input resolution | 96 × 96 | 120 × 120 |
| Tokens | 144 | 225 |
| Hidden dimension | 768 | 256 |
| Heads | 16 | 8 |
| Unimodal depth | 12 optical / 6 SAR | 2 per branch |
| Classes | 8 | 10 by default |

Therefore its results must not be merged into the primary DFC2020 table.

---

## 13. Result Integrity and Reproducibility Caveats

### 13.1 Missing baseline mIoU

[`02_baseline_eval.py`](./02_baseline_eval.py) writes:

```text
results_baseline.txt
```

but that file is not currently present. The persistent baseline measurement available is:

```text
FP32 latency = 663.35 ms/image
```

The FP32 checkpoint size is approximately:

```text
CROMA_base.pt = 741.54 MB
```

The baseline mIoU must be regenerated before it is quoted as a final numerical result.

### 13.2 Result-file overwriting

The result files are written with fixed names, for example:

```python
with open("results_pruned20.txt", "w") as f:
    f.write(...)
```

and:

```python
with open("results_hybrid.txt", "w") as f:
    f.write(...)
```

Running different script generations can overwrite previous measurements.

### 13.3 Subset versus full validation

Several newer scripts evaluate a deterministic 1,000-image subset:

```python
DFCDataset("validation", n=1000)
```

Other scripts evaluate the complete validation set. The paper should report the exact evaluation protocol for every row.

### 13.4 Apparent mismatch in older result summaries

The generated result summary in [`06_generate_results.py`](./06_generate_results.py) expects:

```text
results_baseline.txt
results_quant.txt
results_pruned.txt
results_hybrid.txt
```

However, the current workspace contains:

```text
results_fp32_latency.txt
results_quant.txt
results_pruned20.txt
results_hybrid.txt
```

This indicates that the result-generation script and the current output-file naming convention are not fully synchronized.

### 13.5 Soft pruning does not reduce checkpoint size

Soft pruning zeros selected weights but preserves all original tensors. Consequently, its checkpoint should remain close to the FP32 size:

```text
FP32:       approximately 741.54 MB
Soft-pruned: approximately 741.57 MB
```

The storage reduction is expected only after quantization or a separate sparse-storage/export mechanism.

### 13.6 Structural pruning provides physical reduction

Structural pruning changes the matrix dimensions and therefore reduces the checkpoint size:

```text
FP32:              approximately 741.54 MB
Structural pruned: approximately 690.95 MB
Structural + INT8: approximately 173.54 MB
```

---

## 14. Key Methodology Snippets

## 14.1 2D-ALiBi bias calculation

```python
def get_2dalibi(num_heads, num_patches):
    side = int(math.sqrt(num_patches))
    points = list(
        itertools.product(
            range(side),
            range(side),
        )
    )

    slopes = torch.tensor(
        get_slopes(num_heads)
    ).unsqueeze(1)

    bias_terms = []

    for p1 in points:
        for p2 in points:
            distance = math.sqrt(
                (p1[0] - p2[0]) ** 2 +
                (p1[1] - p2[1]) ** 2
            )
            bias_terms.append(
                -distance * slopes
            )

    bias = torch.cat(
        bias_terms,
        dim=1,
    )

    return bias.view(
        1,
        num_heads,
        num_patches,
        num_patches,
    )
```

## 14.2 Soft L1 attention-head pruning

```python
def prune_self_attn(
    attn,
    prune_ratio=0.20,
    num_heads=16,
):
    weight = attn.to_qkv.weight.data
    dim = weight.shape[1]
    head_dim = dim // num_heads

    scores = []

    for h in range(num_heads):
        sl = slice(
            h * head_dim,
            (h + 1) * head_dim,
        )

        q = weight[sl]
        k = weight[
            dim + sl.start:
            dim + sl.stop
        ]
        v = weight[
            2 * dim + sl.start:
            2 * dim + sl.stop
        ]

        score = (
            q.abs().sum() +
            k.abs().sum() +
            v.abs().sum()
        )
        scores.append(score.item())

    scores = torch.tensor(scores)
    prune_idx = scores.argsort()[
        :int(num_heads * prune_ratio)
    ]

    for h in prune_idx:
        sl = slice(
            h * head_dim,
            (h + 1) * head_dim,
        )

        weight[sl] = 0
        weight[
            dim + sl.start:
            dim + sl.stop
        ] = 0
        weight[
            2 * dim + sl.start:
            2 * dim + sl.stop
        ] = 0
```

## 14.3 Structural head rebuilding

```python
def build_pruned_self_attn(
    old_attn,
    kept_idx,
    dim=768,
    head_dim=48,
):
    n_keep = len(kept_idx)
    new_attn = PrunedAttention(
        dim,
        n_keep,
        head_dim,
    )

    old_qkv = old_attn.to_qkv.weight.data

    rows = []
    rows += [
        old_qkv[
            h * head_dim:
            (h + 1) * head_dim
        ]
        for h in kept_idx
    ]
    rows += [
        old_qkv[
            dim + h * head_dim:
            dim + (h + 1) * head_dim
        ]
        for h in kept_idx
    ]
    rows += [
        old_qkv[
            2 * dim + h * head_dim:
            2 * dim + (h + 1) * head_dim
        ]
        for h in kept_idx
    ]

    new_attn.to_qkv.weight.data = torch.cat(
        rows,
        dim=0,
    )

    cols = [
        old_attn.to_out.weight.data[
            :,
            h * head_dim:
            (h + 1) * head_dim,
        ]
        for h in kept_idx
    ]

    new_attn.to_out.weight.data = torch.cat(
        cols,
        dim=1,
    )

    new_attn.to_out.bias.data.copy_(
        old_attn.to_out.bias.data
    )

    new_attn.input_norm.load_state_dict(
        old_attn.input_norm.state_dict()
    )

    return new_attn
```

## 14.4 Dynamic PyTorch quantization

```python
croma_int8 = torch.quantization.quantize_dynamic(
    croma,
    {nn.Linear},
    dtype=torch.qint8,
)
```

## 14.5 Dynamic ONNX quantization

```python
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)

quantize_dynamic(
    model_input="croma_fp32.onnx",
    model_output="croma_int8.onnx",
    weight_type=QuantType.QUInt8,
)
```

## 14.6 ONNX Runtime single-thread configuration

```python
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = (
    ort.ExecutionMode.ORT_SEQUENTIAL
)

session = ort.InferenceSession(
    onnx_path,
    opts,
    providers=["CPUExecutionProvider"],
)
```

---

## 15. Suggested Paper Reporting Language

The primary architecture can be described as:

> We used the base CROMA multimodal Vision Transformer with two Sentinel-1 SAR channels and twelve Sentinel-2 optical channels. Each 96 × 96 tile was partitioned into non-overlapping 8 × 8 patches, yielding 144 tokens per modality. The base model uses 768-dimensional token embeddings, 16 attention heads, 12 optical encoder layers, 6 SAR encoder layers, and a 6-layer multimodal cross-attention encoder. Spatial relationships were represented using a two-dimensional Euclidean-distance ALiBi relative bias. A frozen CROMA backbone was followed by a trainable 768-to-8 linear probe, with pixel-level labels downsampled to one modal class per 8 × 8 patch.

Dynamic quantization can be described as:

> Dynamic INT8 quantization was applied to all PyTorch `nn.Linear` modules using `torch.quantization.quantize_dynamic` with `torch.qint8` weights. The quantized model was evaluated using a single PyTorch CPU thread, five warm-up passes, and 50 timed inference passes.

Soft pruning can be described as:

> Soft attention-head pruning ranked heads using the L1 magnitude of their Q, K, and V projection slices. The lowest-scoring 20% of heads, corresponding to 3 of 16 heads, were zeroed without changing tensor dimensions.

Structural pruning can be described as:

> Structural pruning selected the 13 most important heads globally from the 16-head attention configuration and rebuilt the QKV and output projections to reduce the attention width from 768 to 624 dimensions before projecting back to the 768-dimensional model space.

Hybrid compression can be described as:

> Hybrid models applied structural or soft pruning first, followed by dynamic INT8 quantization of the resulting linear layers.

---

## 16. Final Reproducibility Checklist

Before using the results as final academic claims:

- [ ] Rerun the FP32 baseline evaluation and save `results_baseline.txt`.
- [ ] Rerun dynamic INT8, soft-pruned, structural, and hybrid models using the same validation split.
- [ ] Record whether the full validation set or the deterministic 1,000-image subset was used.
- [ ] Prevent fixed result filenames from being overwritten by different script generations.
- [ ] Record the exact PyTorch quantized engine.
- [ ] Record CPU model and operating system.
- [ ] Separate model-only size from probe size.
- [ ] Regenerate and capture the 25% primary-model experiment if it is required by the paper.
- [ ] Do not merge the 120×120 representative ONNX results with the 96×96 pretrained CROMA results.
- [ ] Report confidence intervals or repeated-run statistics for latency if the measurements are intended for publication.

