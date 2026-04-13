#!/usr/bin/env python3
"""
OmniClaw Parameter Golf — Modal Training Script v3
Competition-compliant: int6+brotli quantization, <16MB artifact limit

FIXES from v2:
  1. BIGRAM_HASH_SIZE=0 — 16384×8192=134M params was blowing size to 96MB int6!
  2. Data cached on persistent Modal volume (skip re-download)
  3. Longer timeout (training can run >600s; 600s is eval limit, not training)
  4. Clean env vars matching SOTA stack (MLP_MULT=3, depth recurrence, TTT)
  5. Removed BANKING_ENABLED (not in train_gpt_kl.py)
"""
import modal
import subprocess
import os
import sys
import time

app = modal.App("parameter-golf-omniclaw")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "git-lfs")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        "sentencepiece",
        "triton",
        "huggingface_hub",
        "zstandard",
        "brotli",
    )
    .run_commands("git lfs install")
)

vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("parameter-golf-models", create_if_missing=True)

HOURLY_RATE_H100 = 4.89

@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": vol, "/models": MODEL_VOL},
    timeout=7200,   # 2 hours max — 600s is eval wallclock, not training limit
    memory=32768,
)
def train():
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"🐉 OmniClaw Training — {gpu_count}×{torch.cuda.get_device_name(0)}", flush=True)
    print(f"   CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}", flush=True)

    repo_dir = "/workspace/parameter-golf"
    cached_data = "/data/sp8192_cache"

    # Clone/update repo (scripts only, not data)
    if os.path.exists(repo_dir):
        print("📥 Updating repo to latest...", flush=True)
        subprocess.run(["git", "-C", repo_dir, "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", repo_dir, "checkout", "kailean/submission-v1"], check=True)
        subprocess.run(["git", "-C", repo_dir, "reset", "--hard", "origin/kailean/submission-v1"], check=True)
    else:
        print("📥 Cloning repo (scripts only)...", flush=True)
        subprocess.run([
            "git", "clone",
            "-b", "kailean/submission-v1",
            "--depth", "1",
            "https://github.com/kailean/parameter-golf.git",
            repo_dir
        ], check=True)

    # Check if data is cached on persistent volume
    tokenizer_path = f"{cached_data}/datasets/tokenizers/fineweb_8192_bpe.model"
    data_dir = f"{cached_data}/datasets/datasets/fineweb10B_sp8192"

    if not os.path.exists(tokenizer_path):
        print("📥 Downloading SP8192 data to persistent volume (first time only)...", flush=True)
        from huggingface_hub import snapshot_download
        os.makedirs(cached_data, exist_ok=True)
        snapshot_download(
            repo_id="kevclark/parameter-golf",
            repo_type="dataset",
            local_dir=cached_data,
        )
        vol.commit()  # Persist to volume
        print("✅ Data download complete and cached!", flush=True)
    else:
        print("✅ SP8192 data already cached on volume!", flush=True)

    # Verify paths after download
    if not os.path.exists(tokenizer_path):
        print(f"⚠️ Tokenizer not at expected path, searching...", flush=True)
        for root, dirs, files in os.walk(cached_data):
            for f in files:
                if f == "fineweb_8192_bpe.model":
                    tokenizer_path = os.path.join(root, f)
                    print(f"Found tokenizer at: {tokenizer_path}", flush=True)
                    break
        for root, dirs, files in os.walk(cached_data):
            if os.path.basename(root) == "fineweb10B_sp8192" and any(f.endswith('.bin') for f in files):
                data_dir = root
                print(f"Found data dir at: {data_dir}", flush=True)
                break

    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}", flush=True)
        return 1

    shards = [f for f in os.listdir(data_dir) if f.endswith('.bin')]
    print(f"📊 Found {len(shards)} training shards", flush=True)

    # ── Competition Config ──────────────────────────────────────
    # MLP_MULT=3 → ~27M unique params → ~6-7MB int6+brotli (UNDER 16MB)
    # BIGRAM_HASH_SIZE=0 → CRITICAL! 16384×8192=134M params = 96MB int6
    # Depth recurrence L3-5 x2 adds virtual layers without unique params
    # Parallel residuals from L7+ saves compute
    # TTT (doc-independent LoRA) is legal per Issue #1017
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "DATA_PATH": data_dir,
        "TOKENIZER_PATH": tokenizer_path,
        "VOCAB_SIZE": "8192",
        "NUM_LAYERS": "11",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "3",                    # 3x MLP → ~27M params, fits 16MB
        "TIE_EMBEDDINGS": "1",
        "TIED_EMBED_INIT_STD": "0.005",
        "QK_GAIN_INIT": "5.25",
        "ROPE_BASE": "10000.0",
        "LOGIT_SOFTCAP": "30.0",
        "SEED": "1337",
        # Training schedule
        "ITERATIONS": "20000",
        "TRAIN_BATCH_TOKENS": "786432",
        "TRAIN_SEQ_LEN": "2048",
        "WARMUP_STEPS": "20",
        "WARMDOWN_ITERS": "3500",
        "MAX_WALLCLOCK_SECONDS": "3600",     # 1hr training; 600s is eval-only limit
        # Optimizer
        "OPTIMIZER": "muon",
        "MUON_MOMENTUM": "0.99",
        "EMA_DECAY": "0.9965",
        "EMA_START_FRAC": "0.5",
        # QAT
        "QAT_START_FRAC": "0.15",
        "USE_ORTHO_INIT": "1",
        # SOTA architecture
        "SMEAR_ENABLED": "1",
        "ROPE_DIMS": "16",                   # partial RoPE (16/64 dims)
        "LN_SCALE_ENABLED": "1",
        "XSA_LAST_N": "4",
        "USE_GPTQ_LITE": "1",
        # Depth recurrence (L3-5 x2 = ~17 virtual layers, no extra params)
        "DEPTH_RECURRENCE": "1",
        "RECURRENCE_LAYERS": "3,4,5",
        "RECURRENCE_LOOPS": "2",
        # Parallel residuals (GPT-J style, L7+)
        "PARALLEL_RESIDUALS": "1",
        "PARALLEL_RES_START": "7",
        # TTT (doc-independent LoRA — legal per Issue #1017)
        "TTT_ENABLED": "1",
        "TTT_RANK": "4",
        "TTT_LR": "0.001",
        "TTT_STEPS": "3",
        "TTT_ENTROPY_WEIGHT": "1",
        "TTT_DOC_INDEPENDENT": "1",
        # CRITICAL: BigramHash OFF for SP8192 — it's 134M params!
        "BIGRAM_HASH_SIZE": "0",
        # Eval
        "VAL_LOSS_EVERY": "500",
        "VAL_BATCH_SIZE": "524288",
        "TRAIN_LOG_EVERY": "50",
        "EVAL_SEQ_LEN": "2048",
        "EVAL_STRIDE": "64",
        "EVAL_BATCH_SEQS": "32",
        "RUN_ID": "omniclaw_modal_h100_v3",
    }

    script = f"{repo_dir}/train_gpt_kl.py"
    start = time.time()

    process = subprocess.Popen(
        [sys.executable, script],
        env=env,
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    for line in process.stdout:
        print(line.decode(errors='replace'), end='', flush=True)

    process.wait()
    elapsed = time.time() - start
    cost = elapsed / 3600 * HOURLY_RATE_H100
    print(f"\n🐉 Training done! Wallclock: {elapsed:.1f}s, Est. cost: ${cost:.2f}", flush=True)

    if process.returncode != 0:
        print(f"❌ Training failed with exit code {process.returncode}", flush=True)
    else:
        print("✅ Training completed successfully!", flush=True)

    # Persist model artifacts to volume
    MODEL_VOL.commit()

    return process.returncode


@app.local_entrypoint()
def main():
    print("🐉 Starting OmniClaw H100 training v3 on Modal...")
    result = train.remote()
    print(f"Training exited with code: {result}")


if __name__ == "__main__":
    main()