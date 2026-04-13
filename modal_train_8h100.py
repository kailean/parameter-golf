#!/usr/bin/env python3
"""
OmniClaw Parameter Golf — 8×H100 Competition Run
600s wallclock, int6+brotli, <16MB artifact limit

Fixes from previous runs:
  - BIGRAM_HASH_SIZE=0 (was 134M params!)
  - TTT_ENABLED=0 (TTT was hurting: 1.73 vs 1.18 without)
  - 8×H100 with torchrun
  - 600s competition wallclock
  - Data pre-cached on persistent volume
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

HOURLY_RATE_8XH100 = 4.89 * 8  # ~$39.12/hr

@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": vol, "/models": MODEL_VOL},
    timeout=7200,   # 2hr hard cap
    memory=32768,
)
def train():
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"🐉 OmniClaw Training — {gpu_count}×{torch.cuda.get_device_name(0)}", flush=True)
    print(f"   CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}", flush=True)

    repo_dir = "/workspace/parameter-golf"
    cached_data = "/data/sp8192_cache"

    # Clone/update repo
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

    # Check cached data
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
        vol.commit()
        print("✅ Data download complete and cached!", flush=True)
    else:
        print("✅ SP8192 data already cached on volume!", flush=True)

    # Verify paths
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

    # ── Competition Config (8×H100, 600s) ──────────────────────
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "DATA_PATH": data_dir,
        "TOKENIZER_PATH": tokenizer_path,
        "VOCAB_SIZE": "8192",
        "NUM_LAYERS": "11",
        "MODEL_DIM": "640",
        "NUM_HEADS": "10",
        "NUM_KV_HEADS": "5",
        "MLP_MULT": "4",                    # ~52M params → ~14.8MB mixed int8/int6+brotli
        "TIE_EMBEDDINGS": "1",
        "TIED_EMBED_INIT_STD": "0.005",
        "QK_GAIN_INIT": "5.25",
        "ROPE_BASE": "10000.0",
        "LOGIT_SOFTCAP": "30.0",
        "SEED": "1337",
        # Training schedule — competition 600s wallclock
        "ITERATIONS": "20000",
        "TRAIN_BATCH_TOKENS": "786432",
        "TRAIN_SEQ_LEN": "2048",
        "WARMUP_STEPS": "20",
        "WARMDOWN_ITERS": "3500",
        "MAX_WALLCLOCK_SECONDS": "600",     # Competition eval limit
        # Optimizer
        "OPTIMIZER": "muon",
        "MUON_MOMENTUM": "0.99",
        "EMA_DECAY": "0.9965",
        "EMA_START_FRAC": "0.5",
        # QAT — START AT STEP 0 to avoid 350s recompile mid-run!
        "QAT_START_FRAC": "1.0",  # QAT off — rely on GPTQ-lite post-hoc quant
        "USE_ORTHO_INIT": "1",
        # SOTA architecture
        "SMEAR_ENABLED": "1",
        "ROPE_DIMS": "16",
        "LN_SCALE_ENABLED": "1",
        "XSA_LAST_N": "4",
        "USE_GPTQ_LITE": "1",
        # Depth recurrence (L3-5 x2)
        "DEPTH_RECURRENCE": "1",
        "RECURRENCE_LAYERS": "3,4,5",
        "RECURRENCE_LOOPS": "2",
        # Parallel residuals (GPT-J style, L7+)
        "PARALLEL_RESIDUALS": "1",
        "PARALLEL_RES_START": "7",
        # TTT disabled — one variable at a time, test bigger model first
        "TTT_ENABLED": "0",
        "EMBED_BITS": "8",                     # int8 for embeddings, int6 for weights (SOTA approach)
        # CRITICAL: BigramHash OFF for SP8192
        "BIGRAM_HASH_SIZE": "0",
        # Eval — only at the very end (save 30s per eval)
        "VAL_LOSS_EVERY": "99999",
        "VAL_BATCH_SIZE": "524288",
        "TRAIN_LOG_EVERY": "50",
        "EVAL_SEQ_LEN": "2048",
        "EVAL_STRIDE": "64",
        "EVAL_BATCH_SEQS": "32",
        "RUN_ID": "omniclaw_8xh100_v1",
    }

    script = f"{repo_dir}/train_gpt_kl.py"
    start = time.time()

    # Auto-detect GPU count for torchrun
    nproc = max(1, gpu_count)
    print(f"🐉 Launching torchrun with {nproc} GPU(s)...", flush=True)
    process = subprocess.Popen(
        [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            script,
        ],
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
    cost = elapsed / 3600 * HOURLY_RATE_8XH100
    print(f"\n🐉 Training done! Wallclock: {elapsed:.1f}s, Est. cost: ${cost:.2f}", flush=True)

    if process.returncode != 0:
        print(f"❌ Training failed with exit code {process.returncode}", flush=True)
    else:
        print("✅ Training completed successfully!", flush=True)

    # Persist model artifacts
    MODEL_VOL.commit()

    return process.returncode


@app.local_entrypoint()
def main():
    print("🐉 Starting OmniClaw 8×H100 competition run...")
    result = train.remote()
    print(f"Training exited with code: {result}")


if __name__ == "__main__":
    main()