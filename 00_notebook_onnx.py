import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


# ==============================================================================
# 1. Standalone CROMA Multimodal Vision Transformer Architecture
# ==============================================================================
class Spatial2DALiBi(nn.Module):
    """2D Relative Position Encoding with Linear Biases (2D-ALiBi).

    Calculates Euclidean distance penalties between 2D spatial patches.
    """

    def __init__(self, num_heads, grid_size=(15, 15)):
        super().__init__()
        self.num_heads = num_heads
        self.grid_size = grid_size
        # Slopes geometric sequence: 1/2^1, 1/2^2, ..., 1/2^num_heads
        slopes = torch.tensor([1.0 / (2 ** (i + 1)) for i in range(num_heads)])
        self.register_buffer("slopes", slopes.view(num_heads, 1, 1))

    def forward(self, seq_len: int):
        h, w = self.grid_size
        coords = torch.stack(
            torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), dim=-1
        ).float()
        coords = coords.view(-1, 2)  # [L, 2]
        dist = torch.cdist(coords, coords, p=2)  # [L, L]

        # Shape: [num_heads, L, L]
        bias = -dist.unsqueeze(0) * self.slopes
        return bias


class MultiHeadAttentionWithALiBi(nn.Module):
    """Multi-Head Self/Cross-Attention with ALiBi spatial distance penalties."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv, alibi_bias):
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape

        q = (
            self.q_proj(x_q)
            .reshape(B, N_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x_kv)
            .reshape(B, N_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x_kv)
            .reshape(B, N_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if alibi_bias is not None:
            attn_scores = attn_scores + alibi_bias

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N_q, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Standard ViT Encoder Block with LayerNorm, MSA, and FFN."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithALiBi(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, alibi_bias):
        x = x + self.attn(self.norm1(x), self.norm1(x), alibi_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class CROMA_B_Inference(nn.Module):
    """Representative Pipeline for CROMA-B Inference:

    - Sentinel-1 SAR Branch (2 channels)
    - Sentinel-2 Optical Branch (12 channels)
    - Multimodal Cross-Attention Fusion
    - Land-Cover Segmentation Head
    """

    def __init__(self, dim=256, num_heads=8, num_classes=10):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Patch Encoders (8x8 patch size)
        self.sar_patch_embed = nn.Conv2d(2, dim, kernel_size=8, stride=8)
        self.opt_patch_embed = nn.Conv2d(12, dim, kernel_size=8, stride=8)
        self.alibi = Spatial2DALiBi(num_heads=num_heads, grid_size=(15, 15))

        # Unimodal Backbones
        self.sar_encoder = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(2)]
        )
        self.opt_encoder = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(2)]
        )

        # Multimodal Cross-Attention Layer
        self.cross_norm_sar = nn.LayerNorm(dim)
        self.cross_norm_opt = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttentionWithALiBi(dim, num_heads)

        # Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, num_classes)
        )

    def forward(self, sar_img, opt_img):
        # Patch Embeddings: [B, C, H, W] -> [B, L, D]
        x_sar = self.sar_patch_embed(sar_img).flatten(2).transpose(1, 2)
        x_opt = self.opt_patch_embed(opt_img).flatten(2).transpose(1, 2)

        L = x_sar.shape[1]
        alibi_bias = self.alibi(L)

        # Unimodal Encoders
        for blk in self.sar_encoder:
            x_sar = blk(x_sar, alibi_bias)
        for blk in self.opt_encoder:
            x_opt = blk(x_opt, alibi_bias)

        # Multimodal Cross-Attention Fusion
        x_fused = x_sar + self.cross_attn(
            self.cross_norm_sar(x_sar), self.cross_norm_opt(x_opt), alibi_bias
        )

        # Output logits
        logits = self.seg_head(x_fused)
        return logits


# ==============================================================================
# 2. Structured Attention Head Pruning
# ==============================================================================
def prune_attention_heads(model, prune_ratio=0.25):
    """Prunes a percentage of Multi-Head Self-Attention projection weights based

    on L1-norm magnitude score.
    """
    print(
        f"\n[1] Applying Structured Head Pruning (Ratio: {prune_ratio * 100:.0f}%)..."
    )
    pruned_count = 0
    total_heads = 0

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, MultiHeadAttentionWithALiBi):
                num_heads = module.num_heads
                head_dim = module.head_dim

                # Compute L1 norm per head in q_proj
                q_weight = module.q_proj.weight  # [dim, dim]
                head_norms = []
                for h in range(num_heads):
                    head_w = q_weight[
                        h * head_dim : (h + 1) * head_dim, :
                    ]
                    head_norms.append(head_w.abs().sum().item())

                # Identify lowest L1 norm heads
                num_to_prune = max(1, int(num_heads * prune_ratio))
                heads_to_prune = np.argsort(head_norms)[:num_to_prune]

                # Zero out weights for pruned heads
                for h in heads_to_prune:
                    module.q_proj.weight[
                        h * head_dim : (h + 1) * head_dim, :
                    ] = 0.0
                    module.k_proj.weight[
                        h * head_dim : (h + 1) * head_dim, :
                    ] = 0.0
                    module.v_proj.weight[
                        h * head_dim : (h + 1) * head_dim, :
                    ] = 0.0
                    pruned_count += 1
                total_heads += num_heads

    print(
        f" ✓ Pruned {pruned_count}/{total_heads} Attention Heads across all blocks."
    )
    return model


# ==============================================================================
# 3. ONNX Export & Single-Thread CPU Benchmarking
# ==============================================================================
def export_to_onnx(model, dummy_sar, dummy_opt, onnx_path):
    """Exports PyTorch model to ONNX Intermediate Representation."""
    model.eval()
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
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f" ✓ Exported ONNX Model to: {onnx_path} ({size_mb:.2f} MB)")
    return size_mb


def run_int8_quantization(fp32_onnx_path, int8_onnx_path):
    """Applies Post-Training Dynamic INT8 Quantization (W8A8)."""
    print("\n[2] Quantizing ONNX model to INT8 (W8A8)...")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QUInt8,
    )
    size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    print(f" ✓ Quantized INT8 Model saved to: {int8_onnx_path} ({size_mb:.2f} MB)")
    return size_mb


def benchmark_single_threaded_cpu(
    onnx_path, dummy_sar_np, dummy_opt_np, warmup=10, iterations=100
):
    """Simulates nanosatellite single-core CPU execution using ONNX Runtime

    throttled strictly to 1 thread.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        onnx_path, opts, providers=["CPUExecutionProvider"]
    )
    inputs = {"sar_input": dummy_sar_np, "optical_input": dummy_opt_np}

    # Warm-up pass
    for _ in range(warmup):
        session.run(None, inputs)

    # Benchmarking iterations
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    avg_latency = np.mean(times)
    p90_latency = np.percentile(times, 90)
    return avg_latency, p90_latency


# ==============================================================================
# 4. Main Execution Flow
# ==============================================================================
if __name__ == "__main__":
    scratch_dir = "/workspace/scratch"
    os.makedirs(scratch_dir, exist_ok=True)

    print("=" * 70)
    print(" CROMA VISION TRANSFORMER: COMPRESSION & EDGE CPU BENCHMARK PIPELINE ")
    print("=" * 70)

    # Inputs: 120x120 satellite tiles (Sentinel-1 SAR: 2 channels, Sentinel-2 Optical: 12 channels)
    dummy_sar = torch.randn(1, 2, 120, 120)
    dummy_opt = torch.randn(1, 12, 120, 120)
    dummy_sar_np = dummy_sar.numpy()
    dummy_opt_np = dummy_opt.numpy()

    # Step A: Instantiate Baseline Model
    model = CROMA_B_Inference(dim=256, num_heads=8, num_classes=10)
    model.eval()

    # Paths
    fp32_onnx = os.path.join(scratch_dir, "croma_fp32.onnx")
    pruned_fp32_onnx = os.path.join(scratch_dir, "croma_pruned_fp32.onnx")
    int8_onnx = os.path.join(scratch_dir, "croma_int8.onnx")
    hybrid_int8_onnx = os.path.join(scratch_dir, "croma_hybrid_pruned_int8.onnx")

    # Step B: Export FP32 Baseline
    print("\n[A] Exporting FP32 Baseline Model...")
    fp32_size = export_to_onnx(model, dummy_sar, dummy_opt, fp32_onnx)
    fp32_lat, fp32_p90 = benchmark_single_threaded_cpu(
        fp32_onnx, dummy_sar_np, dummy_opt_np
    )

    # Step C: Standalone INT8 Quantization
    int8_size = run_int8_quantization(fp32_onnx, int8_onnx)
    int8_lat, int8_p90 = benchmark_single_threaded_cpu(
        int8_onnx, dummy_sar_np, dummy_opt_np
    )

    # Step D: Apply Structured Head Pruning (25%)
    pruned_model = prune_attention_heads(model, prune_ratio=0.25)
    pruned_size = export_to_onnx(
        pruned_model, dummy_sar, dummy_opt, pruned_fp32_onnx
    )
    pruned_lat, pruned_p90 = benchmark_single_threaded_cpu(
        pruned_fp32_onnx, dummy_sar_np, dummy_opt_np
    )

    # Step E: Hybrid Pruned + INT8 Quantized Model
    hybrid_size = run_int8_quantization(pruned_fp32_onnx, hybrid_int8_onnx)
    hybrid_lat, hybrid_p90 = benchmark_single_threaded_cpu(
        hybrid_int8_onnx, dummy_sar_np, dummy_opt_np
    )

    # Summary Output
    print("\n" + "=" * 70)
    print(" EXPERIMENTAL RESULTS & PARETO BENCHMARK LEDGER ")
    print("=" * 70)
    print(
        f"{'Configuration':<30} | {'Disk Size (MB)':<15} | {'Avg Latency (ms)':<18} | {'P90 Latency (ms)':<18}"
    )
    print("-" * 88)
    print(
        f"{'FP32 Baseline':<30} | {fp32_size:<15.2f} | {fp32_lat:<18.2f} | {fp32_p90:<18.2f}"
    )
    print(
        f"{'Standalone INT8 Quantized':<30} | {int8_size:<15.2f} | {int8_lat:<18.2f} | {int8_p90:<18.2f}"
    )
    print(
        f"{'Structured Pruned (25%)':<30} | {pruned_size:<15.2f} | {pruned_lat:<18.2f} | {pruned_p90:<18.2f}"
    )
    print(
        f"{'Hybrid (Pruned 25% + INT8)':<30} | {hybrid_size:<15.2f} | {hybrid_lat:<18.2f} | {hybrid_p90:<18.2f}"
    )
    print("=" * 70)