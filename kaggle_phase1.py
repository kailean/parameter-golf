#!/usr/bin/env python3
"""
OmniClaw Parameter Golf — Phase 1 Experiment
512d MLP×3 + WD warmdown schedule
Kaggle T4/P100 (FREE, 9hr session limit)
"""
import subprocess
import os
import time
import json

# Install dependencies
print("📦 Installing dependencies...", flush=True)
subprocess.run(["pip", "install", "brotli", "zstandard", "sentencepiece"], check=True, capture_output=True)

# Clone our repo
print("📥 Cloning repo...", flush=True)
if os.path.exists("/kaggle/working/parameter-golf"):
    subprocess.run(["rm", "-rf", "/kaggle/working/parameter-golf"], check=True)
subprocess.run([
    "git", "clone",
    "-b", "kailean/submission-v3",
    "--depth", "1",
    "https://github.com/kailean/parameter-golf.git",
    "/kaggle/working/parameter-golf"
], check=True)

work_dir = "/kaggle/working/parameter-golf"

# Download SP8192 data
data_dir = os.path.join(work_dir, "data", "datasets", "fineweb10B_sp8192")
if not os.path.exists(data_dir):
    print("📥 Downloading SP8192 data (this takes ~5 min)...", flush=True)
    subprocess.run([
        "python3", os.path.join(work_dir, "data", "cached_challenge_fineweb.py"),
        "--variant", "sp8192", "--skip-manifest"
    ], cwd=work_dir, check=True, timeout=600)

# Phase 1 Config: 512d MLP×3 + WD warmdown
env = {
    **os.environ,
    "DATA_PATH": data_dir,
    "TOKENIZER_PATH": os.path.join(work_dir, "data", "tokenizers", "fineweb_8192_bpe.model"),
    # Architecture — 512d MLP×3 (~30M params, guaranteed <16MB)
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
    # Compression
    "EMBED_BITS": "8",
    "USE_GPTQ_LITE": "0",
    "USE_GPTQ_CALIBRATION": "1",
    "GPTQ_CALIBRATION_BATCHES": "64",
    "MATRIX_CLIP_SIGMAS": "12.85",
    "EMBED_CLIP_SIGMAS": "20.0",
    "DELTA_ENCODE_ROWS": "0",
    "BYTE_SHUFFLE_STRIDE": "2",
    "BROTLI_QUALITY": "11",
    # WD warmdown schedule (critical fix)
    "MUON_WEIGHT_DECAY": "0.15",
    "EMBED_WEIGHT_DECAY": "0.15",
    "ADAM_WEIGHT_DECAY": "0.10",
    "WARMDOWN_WD_FRAC": "0.37",
    "LATE_QAT_THRESHOLD": "0.0",
    # Training — fewer steps for T4 (slower GPU)
    "ITERATIONS": "3000",
    "TRAIN_BATCH_TOKENS": "786432",
    "WARMUP_STEPS": "20",
    "WARMDOWN_FRAC": "0.50",
    "VAL_LOSS_EVERY": "500",
    "TRAIN_LOG_EVERY": "50",
    "MAX_WALLCLOCK_SECONDS": "0",
    # TTT
    "TTT_ENABLED": "0",  # Skip TTT for Phase 1 measurement
    # Other
    "EMA_DECAY": "0.9965",
    "EMA_START_FRAC": "0.5",
    "USE_SWA": "0",
    "SEED": "42",
    "DDP_BUCKET_CAP_MB": "64",
}

print("🚀 Starting Phase 1: 512d MLP×3 + WD warmdown", flush=True)
print(f"   Target: {size_bytes} + val_bpb — NOTHING ELSE MATTERS", flush=True)
start = time.time()

proc = subprocess.Popen(
    ["python3", "-u", os.path.join(work_dir, "train_gpt_kl.py")],
    env=env,
    cwd=work_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=0,
)
while True:
    chunk = proc.stdout.read(512)
    if not chunk:
        break
    print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
proc.wait()

elapsed = time.time() - start
print(f"\n⏱ Training took {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

# Phase 1: Report the TWO numbers that matter
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

# Extract best val_bpb
import glob
log_files = sorted(glob.glob(os.path.join(work_dir, "logs", "*.txt")), key=os.path.getmtime, reverse=True)
best_bpb = None
if log_files:
    with open(log_files[0]) as f:
        for line in f:
            if "best_val_bpb" in line:
                parts = line.split("best_val_bpb:")
                if len(parts) > 1:
                    best_bpb = float(parts[1].strip().split()[0])
            elif "val_bpb" in line and "step:" in line:
                # Also extract last val_bpb
                pass

print(f"  Best val_bpb: {best_bpb if best_bpb else 'NOT FOUND'}", flush=True)
print("="*60, flush=True)