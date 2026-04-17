#!/usr/bin/env python3
"""
Minimal Kaggle Phase 1 runner.
This is the NOTEBOOK code — it uses Kaggle's built-in git clone.
"""
import subprocess
import os
import sys
import time

# Step 1: Check environment
print("="*60, flush=True)
print("PHASE 1: 512d MLP×3 + WD Warmdown", flush=True)
print("="*60, flush=True)

# Check GPU
try:
    import torch
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", flush=True)
    else:
        print("⚠️ No GPU! Running on CPU (will be very slow)", flush=True)
except ImportError:
    print("ERROR: PyTorch not installed", flush=True)
    sys.exit(1)

# Step 2: Clone repo
work_dir = "/kaggle/working/parameter-golf"
if os.path.exists(work_dir):
    import shutil
    shutil.rmtree(work_dir)

print("\n📥 Cloning repo...", flush=True)
r = subprocess.run(
    ["git", "clone", "-b", "kailean/submission-v3", "--depth", "1",
     "https://github.com/kailean/parameter-golf.git", work_dir],
    capture_output=True, text=True, timeout=120
)
if r.returncode != 0:
    print(f"Git clone FAILED: {r.stderr[:500]}", flush=True)
    sys.exit(1)
print("✅ Cloned", flush=True)

# Step 3: Install deps
print("\n📦 Installing dependencies...", flush=True)
for pkg in ["brotli", "zstandard", "sentencepiece"]:
    subprocess.run(["pip", "install", "-q", pkg], capture_output=True, timeout=60)
print("✅ Dependencies installed", flush=True)

# Step 4: Download data
data_dir = os.path.join(work_dir, "data", "datasets", "fineweb10B_sp8192")
if not os.path.exists(data_dir):
    print("\n📥 Downloading SP8192 data...", flush=True)
    r = subprocess.run(
        ["python3", os.path.join(work_dir, "data", "cached_challenge_fineweb.py"),
         "--variant", "sp8192", "--skip-manifest"],
        cwd=work_dir, capture_output=True, text=True, timeout=600
    )
    if r.returncode != 0:
        print(f"Data download FAILED: {r.stderr[:500]}", flush=True)
        sys.exit(1)
    print("✅ Data downloaded", flush=True)

shards = [f for f in os.listdir(data_dir) if f.endswith(".bin") and "train" in f]
print(f"  Train shards: {len(shards)}", flush=True)

# Step 5: Train!
env = {
    **os.environ,
    "DATA_PATH": data_dir,
    "TOKENIZER_PATH": os.path.join(work_dir, "data", "tokenizers", "fineweb_8192_bpe.model"),
    "VOCAB_SIZE": "8192",
    "MODEL_DIM": "512",
    "NUM_LAYERS": "13",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_MULT": "3",
    "TIE_EMBEDDINGS": "1",
    "TIED_EMBED_INIT_STD": "0.005",
    "LOGIT_SOFTCAP": "30.0",
    "ROPE_BASE": "10000.0",
    "QK_GAIN_INIT": "5.0",
    "BIGRAM_HASH_SIZE": "0",
    "USE_ORTHO_INIT": "1",
    "SMEAR_ENABLED": "1",
    "XSA_LAST_N": "13",
    "ROPE_DIMS": "16",
    "LN_SCALE_ENABLED": "1",
    "DEPTH_RECURRENCE": "1",
    "RECURRENCE_LAYERS": "3,4,5",
    "RECURRENCE_LOOPS": "2",
    "PARALLEL_RESIDUALS": "1",
    "PARALLEL_RES_START": "7",
    "PARALLEL_FINAL_LANE_MEAN": "1",
    "LEAKY_RELU_SLOPE": "0.5",
    "EMBED_BITS": "8",
    "USE_GPTQ_LITE": "0",
    "USE_GPTQ_CALIBRATION": "1",
    "GPTQ_CALIBRATION_BATCHES": "64",
    "MATRIX_CLIP_SIGMAS": "12.85",
    "EMBED_CLIP_SIGMAS": "20.0",
    "DELTA_ENCODE_ROWS": "0",
    "BYTE_SHUFFLE_STRIDE": "2",
    "BROTLI_QUALITY": "11",
    "MUON_WEIGHT_DECAY": "0.15",
    "EMBED_WEIGHT_DECAY": "0.15",
    "ADAM_WEIGHT_DECAY": "0.10",
    "WARMDOWN_WD_FRAC": "0.37",
    "LATE_QAT_THRESHOLD": "0.0",
    "ITERATIONS": "2000",
    "TRAIN_BATCH_TOKENS": "393216",
    "WARMUP_STEPS": "20",
    "WARMDOWN_FRAC": "0.50",
    "VAL_LOSS_EVERY": "500",
    "TRAIN_LOG_EVERY": "50",
    "MAX_WALLCLOCK_SECONDS": "0",
    "TTT_ENABLED": "0",
    "EMA_DECAY": "0.9965",
    "EMA_START_FRAC": "0.5",
    "USE_SWA": "0",
    "SEED": "42",
    "DDP_BUCKET_CAP_MB": "64",
}

print("\n🚀 Starting training...", flush=True)
start = time.time()

proc = subprocess.Popen(
    ["python3", "-u", os.path.join(work_dir, "train_gpt_kl.py")],
    env=env, cwd=work_dir,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
)
while True:
    chunk = proc.stdout.read(512)
    if not chunk:
        break
    print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
proc.wait()

elapsed = time.time() - start
print(f"\n⏱ Took {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

# Step 6: Report results
print("\n" + "="*60, flush=True)
print("PHASE 1 RESULTS — THE ONLY NUMBERS THAT MATTER", flush=True)
print("="*60, flush=True)

model_path = os.path.join(work_dir, "final_model.int6.brotli.ptz")
if os.path.exists(model_path):
    size_bytes = os.path.getsize(model_path)
    code_path = os.path.join(work_dir, "train_gpt_kl.py")
    cs_path = os.path.join(work_dir, "custom_serializer.py")
    code_bytes = len(open(code_path).read().encode()) + len(open(cs_path).read().encode())
    total = size_bytes + code_bytes
    fits = total < 16_000_000
    print(f"  Compressed size: {total:,} bytes ({total/1e6:.2f}MB)", flush=True)
    print(f"  FITS 16MB? {'YES ✅' if fits else 'NO ❌'}", flush=True)
else:
    print("  ❌ Model file not found", flush=True)

import glob
best_bpb = None
for lf in sorted(glob.glob(os.path.join(work_dir, "logs", "*.txt")), key=os.path.getmtime, reverse=True):
    with open(lf) as f:
        for line in f:
            if "best_val_bpb:" in line:
                parts = line.split("best_val_bpb:")[1].strip().split()
                try:
                    best_bpb = float(parts[0])
                except:
                    pass
                break
    if best_bpb:
        break

print(f"  Best val_bpb: {best_bpb if best_bpb else 'NOT FOUND'}", flush=True)
print("="*60, flush=True)