#!/usr/bin/env python3
"""
OmniClaw Parameter Golf — Compression-Safe SOTA v2
DIM=384, 13L, MLP×3, ~21M params → fits under 16MB even at 0.51 ratio
Aggressive weight decay (0.15) for compression, all SOTA techniques
1×H100 on Modal (williguse account)
"""
import modal
import subprocess
import os
import time

app = modal.App("parameter-golf-sota-v2")

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
        print(f"🐉 OmniClaw SOTA v2 (Compression-Safe) — {gpu_count}×{self.gpu_name}", flush=True)
        print(f"   Config: 13L, DIM=384, MLP×3, 6Q/3KV, WD=0.15, aggressive compression", flush=True)

        repo_dir = "/workspace/parameter-golf"

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

        train_shards = len([f for f in os.listdir(data_dir) if "train" in f and f.endswith(".bin")])
        val_shards = len([f for f in os.listdir(data_dir) if "val" in f and f.endswith(".bin")])
        print(f"📊 Train shards: {train_shards}, Val shards: {val_shards}", flush=True)

        # COMPRESSION-SAFE SOTA v2: ~21M params, guaranteed <16MB
        # Key changes: DIM=384, MLP=3, WD=0.15 (aggressive for compression)
        env = {
            **os.environ,
            "DATA_PATH": data_dir,
            "TOKENIZER_PATH": os.path.join(tokenizer_dir, "fineweb_8192_bpe.model") if os.path.exists(tokenizer_dir) else os.path.join(cached_tokenizers, "fineweb_8192_bpe.model"),
            # Architecture — smaller but deeper
            "VOCAB_SIZE": "8192",
            "MODEL_DIM": "384",
            "NUM_LAYERS": "13",
            "NUM_HEADS": "6",
            "NUM_KV_HEADS": "3",
            "MLP_MULT": "3",
            "TIE_EMBEDDINGS": "1",
            "TIED_EMBED_INIT_STD": "0.005",
            "LOGIT_SOFTCAP": "30.0",
            "ROPE_BASE": "10000.0",
            "QK_GAIN_INIT": "5.0",
            "BIGRAM_HASH_SIZE": "0",
            "USE_ORTHO_INIT": "1",
            # Attention
            "SMEAR_ENABLED": "1",
            "XSA_LAST_N": "13",
            "ROPE_DIMS": "16",
            "LN_SCALE_ENABLED": "1",
            # Depth recurrence
            "DEPTH_RECURRENCE": "1",
            "RECURRENCE_LAYERS": "3,4,5",
            "RECURRENCE_LOOPS": "2",
            # Parallel residuals
            "PARALLEL_RESIDUALS": "1",
            "PARALLEL_RES_START": "7",
            "PARALLEL_FINAL_LANE_MEAN": "1",
            # Activation
            "LEAKY_RELU_SLOPE": "0.5",
            # Compression — aggressive WD for better entropy
            "EMBED_BITS": "8",
            "USE_GPTQ_LITE": "0",
            "USE_GPTQ_CALIBRATION": "1",
            "GPTQ_CALIBRATION_BATCHES": "64",
            "MATRIX_CLIP_SIGMAS": "12.85",
            "EMBED_CLIP_SIGMAS": "20.0",
            "DELTA_ENCODE_ROWS": "0",
            "BYTE_SHUFFLE_STRIDE": "2",
            "BROTLI_QUALITY": "11",
            # CRITICAL: Aggressive weight decay for compression
            "MUON_WEIGHT_DECAY": "0.15",    # Was 0.095, now 0.15 for compression
            "EMBED_WEIGHT_DECAY": "0.15",    # Was 0.085, now 0.15 for compression
            "ADAM_WEIGHT_DECAY": "0.10",     # Was 0.04, now 0.10 for compression
            "LATE_QAT_THRESHOLD": "0.0",
            # Training schedule
            "ITERATIONS": "2000",
            "TRAIN_BATCH_TOKENS": "786432",
            "WARMUP_STEPS": "20",
            "WARMDOWN_FRAC": "0.72",
            "VAL_LOSS_EVERY": "500",
            "TRAIN_LOG_EVERY": "50",
            "MAX_WALLCLOCK_SECONDS": "0",  # No cap
            # TTT
            "TTT_ENABLED": "1",
            "TTT_RANK": "96",
            "TTT_LR": "0.0001",
            "TTT_STEPS": "3",
            "TTT_GRAD_STEPS": "1",
            "TTT_CHUNK": "32",
            "TTT_BATCH": "64",
            "TTT_BETA1": "0.0",
            "TTT_BETA2": "0.999",
            "TTT_WEIGHT_DECAY": "0.5",
            "TTT_DOC_INDEPENDENT": "1",
            # Other
            "EMA_DECAY": "0.9965",
            "EMA_START_FRAC": "0.5",
            "USE_SWA": "0",
            "SEED": "42",
            "DDP_BUCKET_CAP_MB": "64",
        }

        print("🚀 Starting SOTA v2 (compression-safe) training...", flush=True)
        print(f"   Key: 13L×384d MLP3, 6Q/3KV, WD=0.15, GPTQ cal, embed_bits=8", flush=True)
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
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
        proc.wait()

        elapsed = time.time() - start
        print(f"\n⏱ Training took {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
        print(f"Exit code: {proc.returncode}", flush=True)

        # Compression verification
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

        # Copy artifacts
        import shutil, glob
        for fname in ["final_model.int6.brotli.ptz", "final_model.int6.ptz", "final_model.int8.ptz", "final_model.pt"]:
            src = os.path.join(work_dir, fname)
            if os.path.exists(src):
                dst = os.path.join("/models", f"sota_v2_{fname}")
                shutil.copy2(src, dst)
                print(f"  Saved → /models/sota_v2_{fname} ({os.path.getsize(dst):,} bytes)", flush=True)
        MODEL_VOL.reload()

        log_files = sorted(glob.glob(os.path.join(work_dir, "logs", "*.txt")), key=os.path.getmtime, reverse=True)
        if log_files:
            print(f"\n📋 Key results:", flush=True)
            with open(log_files[0]) as f:
                for line in f:
                    if any(k in line for k in ["val_bpb", "serialized_int6_brotli", "final_", "total_params", "ttt_val", "FITS"]):
                        print(f"  {line.strip()}", flush=True)

        return proc.returncode


@app.local_entrypoint()
def main():
    trainer = Trainer()
    trainer.train.remote()