#!/usr/bin/env python3
"""
OmniClaw Parameter Golf — 1×H100 640d Competition Training
Custom serializer + no byte shuffle + embed_bits=8
Target: 2000 steps, verify compression under 16MB
Uses modal.Cls for persistent background execution.
"""
import modal
import subprocess
import os
import time

app = modal.App("parameter-golf-640d-h100")

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


@app.cls(
    image=image,
    gpu="H100",
    volumes={"/data": vol, "/models": MODEL_VOL},
    timeout=7200,
    memory=32768,
)
class Trainer:
    @modal.enter()
    def setup(self):
        import torch
        self.gpu_name = torch.cuda.get_device_name(0)
        print(f"🐉 GPU: {self.gpu_name}", flush=True)

    @modal.method()
    def train(self):
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"🐉 OmniClaw 1×H100 — {gpu_count}×{self.gpu_name}", flush=True)
        print(f"   Config: 640d MLPx4, emb8, custom serializer, no byte shuffle", flush=True)

        repo_dir = "/workspace/parameter-golf"

        # Clone repo fresh
        if os.path.exists(repo_dir):
            subprocess.run(["rm", "-rf", repo_dir], check=True)
        print("📥 Cloning repo...", flush=True)
        subprocess.run([
            "git", "clone",
            "-b", "kailean/submission-v3",
            "--depth", "1",
            "https://github.com/kailean/parameter-golf.git",
            repo_dir
        ], check=True)

        work_dir = repo_dir

        # Symlink cached data
        cached_datasets = "/data/sp8192_cache/datasets/datasets"
        cached_tokenizers = "/data/sp8192_cache/datasets/tokenizers"
        data_dir = os.path.join(work_dir, "data", "datasets", "fineweb10B_sp8192")
        tokenizer_dir = os.path.join(work_dir, "data", "tokenizers")

        os.makedirs(os.path.join(work_dir, "data", "datasets"), exist_ok=True)
        if not os.path.exists(data_dir):
            if os.path.exists(cached_datasets + "/fineweb10B_sp8192"):
                os.symlink(cached_datasets + "/fineweb10B_sp8192", data_dir)
                print(f"📋 Symlinked cached SP8192 data", flush=True)
            else:
                print("📥 Downloading SP8192 data...", flush=True)
                subprocess.run([
                    "python3", os.path.join(work_dir, "data", "cached_challenge_fineweb.py"),
                    "--variant", "sp8192", "--skip-manifest"
                ], cwd=work_dir, check=True, timeout=600)

        if not os.path.exists(tokenizer_dir) and os.path.exists(cached_tokenizers):
            os.symlink(cached_tokenizers, tokenizer_dir)
            print(f"📋 Symlinked cached tokenizers", flush=True)

        # Verify data
        train_shards = len([f for f in os.listdir(data_dir) if "train" in f and f.endswith(".bin")])
        val_shards = len([f for f in os.listdir(data_dir) if "val" in f and f.endswith(".bin")])
        print(f"📊 Train shards: {train_shards}, Val shards: {val_shards}", flush=True)

        # Environment
        env = {
            **os.environ,
            "DATA_PATH": data_dir,
            "TOKENIZER_PATH": os.path.join(tokenizer_dir, "fineweb_8192_bpe.model") if os.path.exists(tokenizer_dir) else os.path.join(cached_tokenizers, "fineweb_8192_bpe.model"),
            "VOCAB_SIZE": "8192",
            "MODEL_DIM": "640",
            "NUM_LAYERS": "11",
            "NUM_HEADS": "10",
            "NUM_KV_HEADS": "5",
            "MLP_MULT": "4",
            "TIE_EMBEDDINGS": "1",
            "TIED_EMBED_INIT_STD": "0.005",
            "LOGIT_SOFTCAP": "30.0",
            "ROPE_BASE": "10000.0",
            "QK_GAIN_INIT": "5.25",
            "BIGRAM_HASH_SIZE": "0",
            "USE_ORTHO_INIT": "1",
            "SMEAR_ENABLED": "1",
            "XSA_LAST_N": "4",
            "ROPE_DIMS": "16",
            "LN_SCALE_ENABLED": "1",
            "DEPTH_RECURRENCE": "1",
            "RECURRENCE_LAYERS": "3,4,5",
            "RECURRENCE_LOOPS": "2",
            "PARALLEL_RESIDUALS": "1",
            "PARALLEL_RES_START": "7",
            "EMBED_BITS": "8",
            "ITERATIONS": "2000",
            "TRAIN_BATCH_TOKENS": "786432",
            "WARMUP_STEPS": "20",
            "WARMDOWN_ITERS": "700",
            "VAL_LOSS_EVERY": "200",
            "TRAIN_LOG_EVERY": "50",
            "MAX_WALLCLOCK_SECONDS": "0",
            "LATE_QAT_THRESHOLD": "0.0",
            "USE_GPTQ_LITE": "1",
            "OPTIMIZER": "muon",
            "MUON_MOMENTUM": "0.99",
            "EMA_DECAY": "0.9965",
            "EMA_START_FRAC": "0.5",
            "USE_SWA": "0",
            "TTT_ENABLED": "0",
            "SEED": "42",
        }

        print("🚀 Starting training...", flush=True)
        start = time.time()

        proc = subprocess.Popen(
            ["python3", os.path.join(work_dir, "train_gpt_kl.py")],
            env=env,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Stream output line by line
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        result = proc

        elapsed = time.time() - start
        print(f"\n⏱ Training took {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
        print(f"Exit code: {result.returncode}", flush=True)

        # Verify compression
        print("\n📦 Compression verification...", flush=True)
        model_path = os.path.join(work_dir, "final_model.int6.brotli.ptz")
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            code_path = os.path.join(work_dir, "train_gpt_kl.py")
            cs_path = os.path.join(work_dir, "custom_serializer.py")
            code_bytes = len(open(code_path).read().encode()) + len(open(cs_path).read().encode())
            total = size + code_bytes
            fits = total < 16_000_000
            print(f"  Model: {size:,} bytes ({size/1e6:.2f}MB)", flush=True)
            print(f"  Code:  {code_bytes:,} bytes ({code_bytes/1e6:.2f}MB)", flush=True)
            print(f"  TOTAL: {total:,} bytes ({total/1e6:.2f}MB)", flush=True)
            print(f"  FITS 16MB? {'YES ✅' if fits else 'NO ❌'}", flush=True)

        # Copy artifacts to volume
        import shutil
        import glob
        for fname in ["final_model.int6.brotli.ptz", "final_model.int6.ptz", "final_model.int8.ptz", "final_model.pt"]:
            src = os.path.join(work_dir, fname)
            if os.path.exists(src):
                dst = os.path.join("/models", fname)
                shutil.copy2(src, dst)
                print(f"  Saved {fname} → /models ({os.path.getsize(dst):,} bytes)", flush=True)
        MODEL_VOL.reload()

        # Check training logs for val_bpb
        log_files = sorted(glob.glob(os.path.join(work_dir, "logs", "*.txt")), key=os.path.getmtime, reverse=True)
        if log_files:
            latest_log = log_files[0]
            print(f"\n📋 Latest log: {latest_log}", flush=True)
            with open(latest_log) as f:
                for line in f:
                    if "val_bpb" in line or "serialized_int6_brotli" in line or "final_" in line:
                        print(f"  {line.strip()}", flush=True)

        return result.returncode


@app.local_entrypoint()
def main():
    trainer = Trainer()
    trainer.train.remote()