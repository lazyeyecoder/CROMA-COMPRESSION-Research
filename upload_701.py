from huggingface_hub import HfApi
import os

api = HfApi()
REPO_ID = "lazyeyecoder/croma-compression-research"

LOCAL_DIR = r"C:\Users\Admin\Downloads\Research_Hons_23\CROMA-COMPRESSION-Research\PTs"

files_to_upload = [
    "croma_hybrid_structural_full.pt",
    "croma_structural_pruned_full.pt",
    "croma_pruned_20_full.pt",
    "CROMA_INT8_Dynamic_full.pt",
    "probe_baseline_full.pt",
    "probe_baseline.pt",
    "dfc_val_features_full.pt",
    "dfc_train_features_full.pt",
    "CROMA_base.pt",
]

for fname in files_to_upload:
    local_path = os.path.join(LOCAL_DIR, fname)
    if not os.path.exists(local_path):
        print(f"SKIP (not found): {local_path}")
        continue
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    repo_path = f"701/{fname}"
    print(f"Uploading {fname} ({size_mb:.1f} MiB) -> {repo_path}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="model",
    )

print("\nDone. Verify at:")
print(f"https://huggingface.co/{REPO_ID}/tree/main/701")