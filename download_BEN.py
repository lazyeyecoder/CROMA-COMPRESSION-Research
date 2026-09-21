from datasets import load_dataset, Dataset

# ==========================================
# 1. GET 10% OF THE TRAINING SPLIT (~26,000)
# ==========================================
print("Connecting to Hugging Face for Training Data (Streaming Mode)...")
train_stream = load_dataset(
    "GFM-Bench/BigEarthNet", 
    split="train", 
    streaming=True,
    trust_remote_code=True
)

train_samples = []
target_train_samples = 26000

print(f"Extracting {target_train_samples} training images...")
for i, sample in enumerate(train_stream):
    train_samples.append(sample)
    if (i + 1) % 5000 == 0:
        print(f"Grabbed {i + 1} / {target_train_samples} training samples...")
    if i + 1 >= target_train_samples:
        break

print("Saving training set locally to your hard drive...")
local_train = Dataset.from_list(train_samples)
local_train.save_to_disk("bigearthnet_10pct_train")
print("Saved training set to 'bigearthnet_10pct_train'\n")
print("-" * 50 + "\n")


# ==========================================
# 2. GET A CLEAN VALIDATION SPLIT (~3,000)
# ==========================================
print("Connecting to Hugging Face for Validation Data (Streaming Mode)...")
val_stream = load_dataset(
    "GFM-Bench/BigEarthNet", 
    split="val", 
    streaming=True,
    trust_remote_code=True
)

val_samples = []
target_val_samples = 3000

print(f"Extracting {target_val_samples} validation images...")
for i, sample in enumerate(val_stream):
    val_samples.append(sample)
    if (i + 1) % 1000 == 0:
        print(f"Grabbed {i + 1} / {target_val_samples} validation samples...")
    if i + 1 >= target_val_samples:
        break

print("Saving validation set locally to your hard drive...")
local_val = Dataset.from_list(val_samples)
local_val.save_to_disk("bigearthnet_val_local")
print("Saved validation set to 'bigearthnet_val_local'\n")

print("Done! All set!")