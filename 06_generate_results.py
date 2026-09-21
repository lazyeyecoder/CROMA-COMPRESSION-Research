def load_res(fname):
  with open(fname) as f:
    parts = f.read().strip().split(",")
    return float(parts[0]), float(parts[1]), float(parts[2])


m_base, l_base, s_base = load_res("results_baseline.txt")
m_quant, l_quant, s_quant = load_res("results_quant.txt")
m_prune, l_prune, s_prune = load_res("results_pruned.txt")
m_hybr, l_hybr, s_hybr = load_res("results_hybrid.txt")

print("\n" + "=" * 80)
print("       FINAL RESEARCH RESULTS: CROMA MODEL COMPRESSION (DFC2020)")
print("=" * 80)
print(
    f"{'Model Variant':<22} | {'Precision':<10} | {'mIoU (DFC)':<10} |"
    f" {'Latency (ms)':<12} | {'Size (MB)':<10} | {'Speedup':<8}"
)
print("-" * 80)
print(
    f"{'1. FP32 Baseline':<22} | {'FP32':<10} | {m_base:<10.4f} |"
    f" {l_base:<12.2f} | {s_base:<10.2f} | 1.0x"
)
print(
    f"{'2. INT8 Quantized':<22} | {'INT8':<10} | {m_quant:<10.4f} |"
    f" {l_quant:<12.2f} | {s_quant:<10.2f} |"
    f" {l_base/l_quant:<8.1f}x"
)
print(
    f"{'3. Pruned 20%':<22} | {'FP32':<10} | {m_prune:<10.4f} |"
    f" {l_prune:<12.2f} | {s_prune:<10.2f} |"
    f" {l_base/l_prune:<8.1f}x"
)
print(
    f"{'4. Hybrid (P20+INT8)':<22} | {'INT8':<10} | {m_hybr:<10.4f} |"
    f" {l_hybr:<12.2f} | {s_hybr:<10.2f} |"
    f" {l_base/l_hybr:<8.1f}x"
)
print("=" * 80 + "\n")