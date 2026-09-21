import time, torch
from use_croma import PretrainedCROMA

def main():
    croma = PretrainedCROMA(pretrained_path="CROMA_base.pt", size="base", modality="both", image_resolution=96)
    croma.eval()

    torch.set_num_threads(1)
    dummy_s1 = torch.randn(1, 2, 96, 96)
    dummy_s2 = torch.randn(1, 12, 96, 96)

    with torch.no_grad():
        for _ in range(5):
            croma(SAR_images=dummy_s1, optical_images=dummy_s2)
        t0 = time.perf_counter()
        for _ in range(50):
            croma(SAR_images=dummy_s1, optical_images=dummy_s2)
        t1 = time.perf_counter()

    latency_ms = (t1 - t0) / 50 * 1000
    print(f"FP32 baseline latency: {latency_ms:.2f} ms/image")

    with open("results_fp32_latency.txt", "w") as f:
        f.write(f"{latency_ms:.2f}")

if __name__ == "__main__":
    main()