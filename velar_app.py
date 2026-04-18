import velar

app = velar.App("parameter-golf-train")

image = velar.Image.from_registry(
    "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
).pip_install(
    "brotli", "zstandard", "sentencepiece", "numpy",
)


@app.function(gpu="A100", image=image, timeout=7200)
def train_model():
    import subprocess
    import os
    import time
    import torch

    print(f"PyTorch: {torch.__version__} CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", flush=True)

    work_dir = "/tmp/parameter-golf"
    if os.path.exists(work_dir):
        import shutil
        shutil.rmtree(work_dir)

    print("Cloning repo...", flush=True)
    subprocess.run(
        ["git", "clone", "-b", "kailean/submission-v3", "--depth", "1",
         "https://github.com/kailean/parameter-golf.git", work_dir],
        check=True, timeout=120,
    )

    data_dir = os.path.join(work_dir, "data", "datasets", "fineweb10B_sp8192")
    if not os.path.exists(data_dir):
        print("Downloading SP8192 data...", flush=True)
        subprocess.run(
            ["python3", os.path.join(work_dir, "data", "cached_challenge_fineweb.py"),
             "--variant", "sp8192", "--skip-manifest"],
            cwd=work_dir, check=True, timeout=600,
        )

    env = dict(os.environ,
        DATA_PATH=data_dir,
        TOKENIZER_PATH=os.path.join(work_dir, "data", "tokenizers", "fineweb_8192_bpe.model"),
        VOCAB_SIZE="8192", MODEL_DIM="512", NUM_LAYERS="13", NUM_HEADS="8",
        NUM_KV_HEADS="4", MLP_MULT="3", TIE_EMBEDDINGS="1",
        TIED_EMBED_INIT_STD="0.005", LOGIT_SOFTCAP="30.0", ROPE_BASE="10000.0",
        QK_GAIN_INIT="5.0", BIGRAM_HASH_SIZE="0", USE_ORTHO_INIT="1",
        SMEAR_ENABLED="1", XSA_LAST_N="13", ROPE_DIMS="16",
        LN_SCALE_ENABLED="1", DEPTH_RECURRENCE="1",
        RECURRENCE_LAYERS="3,4,5", RECURRENCE_LOOPS="2",
        PARALLEL_RESIDUALS="1", PARALLEL_RES_START="7",
        PARALLEL_FINAL_LANE_MEAN="1", LEAKY_RELU_SLOPE="0.5",
        EMBED_BITS="8", USE_GPTQ_CALIBRATION="1",
        GPTQ_CALIBRATION_BATCHES="64", MATRIX_CLIP_SIGMAS="12.85",
        EMBED_CLIP_SIGMAS="20.0", DELTA_ENCODE_ROWS="0",
        BYTE_SHUFFLE_STRIDE="2", BROTLI_QUALITY="11",
        GRADUATED_QAT="1", GRADUATED_QAT_INT8_FRAC="0.30",
        GRADUATED_QAT_INT6_FRAC="0.50",
        ENTROPY_REG_WEIGHT="0.001", ENTROPY_REG_WARMUP_FRAC="0.50",
        SELF_GEN_GPTQ_CALIBRATION="1",
        MUON_WEIGHT_DECAY="0.15", EMBED_WEIGHT_DECAY="0.15",
        ADAM_WEIGHT_DECAY="0.10", WARMDOWN_WD_FRAC="0.37",
        LATE_QAT_THRESHOLD="0.0",
        ITERATIONS="5000", TRAIN_BATCH_TOKENS="786432",
        WARMUP_STEPS="20", WARMDOWN_FRAC="0.50",
        VAL_LOSS_EVERY="500", TRAIN_LOG_EVERY="50",
        MAX_WALLCLOCK_SECONDS="0",
        TTT_ENABLED="0", EMA_DECAY="0.9965",
        EMA_START_FRAC="0.5", USE_SWA="0", SEED="42",
    )

    print("Starting training...", flush=True)
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
    print(f"\nTraining took {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    model_path = os.path.join(work_dir, "final_model.int6.brotli.ptz")
    total = None
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        code_bytes = (
            len(open(os.path.join(work_dir, "train_gpt_kl.py")).read().encode())
            + len(open(os.path.join(work_dir, "custom_serializer.py")).read().encode())
        )
        total = size_bytes + code_bytes
        fits = total < 16_000_000
        print(f"Compressed: {total:,} bytes ({total/1e6:.2f}MB) FITS: {fits}", flush=True)

    import glob
    best_bpb = None
    for lf in sorted(glob.glob(os.path.join(work_dir, "logs", "*.txt")), key=os.path.getmtime, reverse=True):
        with open(lf) as f:
            for line in f:
                if "best_val_bpb:" in line:
                    try:
                        best_bpb = float(line.split("best_val_bpb:")[1].strip().split()[0])
                    except Exception:
                        pass
                    break
        if best_bpb:
            break

    print(f"Best val_bpb: {best_bpb}", flush=True)
    return {"best_bpb": best_bpb, "total_bytes": total}