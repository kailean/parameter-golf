#!/usr/bin/env python3
"""
OmniClaw Parameter Golf — Modal Training Script v2
Competition-compliant: int8+zlib quantization, 16MB artifact limit
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
    timeout=3600,
    memory=32768,
)
def train():
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"🐉 OmniClaw Training — {gpu_count}×{torch.cuda.get_device_name(0)}", flush=True)
    print(f"   CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}", flush=True)
    
    repo_dir = "/workspace/parameter-golf"
    if os.path.exists(repo_dir):
        print("📥 Updating repo to latest...", flush=True)
        subprocess.run(["git", "-C", repo_dir, "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", repo_dir, "checkout", "kailean/submission-v1"], check=True)
        subprocess.run(["git", "-C", repo_dir, "reset", "--hard", "origin/kailean/submission-v1"], check=True)
    else:
        print("📥 Cloning repo...", flush=True)
        subprocess.run([
            "git", "clone",
            "-b", "kailean/submission-v1",
            "https://github.com/kailean/parameter-golf.git",
            repo_dir
        ], check=True)
    
    data_dir = f"{repo_dir}/datasets/datasets/fineweb10B_sp8192"
    tokenizer_path = f"{repo_dir}/datasets/tokenizers/fineweb_8192_bpe.model"
    if not os.path.exists(tokenizer_path):
        print("📥 Downloading SP8192 data from HuggingFace dataset...", flush=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="kevclark/parameter-golf",
            repo_type="dataset",
            local_dir=repo_dir,
        )
        print("✅ Data download complete!", flush=True)
    else:
        print("✅ SP8192 data already cached!", flush=True)
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}", flush=True)
        return 1
    
    shards = [f for f in os.listdir(data_dir) if f.endswith('.bin')]
    print(f"📊 Found {len(shards)} training shards", flush=True)
    
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
        "MLP_MULT": "4",
        "TIE_EMBEDDINGS": "1",
        "TIED_EMBED_INIT_STD": "0.005",
        "QK_GAIN_INIT": "5.25",
        "ROPE_BASE": "10000.0",
        "LOGIT_SOFTCAP": "30.0",
        "SEED": "1337",
        "ITERATIONS": "20000",
        "TRAIN_BATCH_TOKENS": "786432",
        "TRAIN_SEQ_LEN": "2048",
        "WARMUP_STEPS": "20",
        "WARMDOWN_ITERS": "3500",
        "MAX_WALLCLOCK_SECONDS": "600",
        "OPTIMIZER": "muon",
        "MUON_MOMENTUM": "0.99",
        "EMA_DECAY": "0.9965",
        "QAT_START_FRAC": "0.15",
        "DEPTH_RECURRENCE": "1",
        "RECURRENCE_LAYERS": "3,4,5",
        "RECURRENCE_LOOPS": "2",
        "PARALLEL_RESIDUALS": "1",
        "PARALLEL_RES_START": "7",
        "TTT_ENABLED": "1",
        "TTT_RANK": "4",
        "TTT_LR": "0.001",
        "TTT_STEPS": "3",
        "TTT_ENTROPY_WEIGHT": "1",
        "TTT_DOC_INDEPENDENT": "1",
        "BANKING_ENABLED": "1",
        "VAL_LOSS_EVERY": "500",
        "VAL_BATCH_SIZE": "524288",
        "TRAIN_LOG_EVERY": "50",
        "RUN_ID": "omniclaw_modal_h100_v2",
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
    
    return process.returncode


@app.local_entrypoint()
def main():
    print("🐉 Starting OmniClaw H100 training v2 on Modal...")
    result = train.remote()
    print(f"Training exited with code: {result}")


if __name__ == "__main__":
    main()