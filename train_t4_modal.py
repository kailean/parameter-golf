"""Parameter Golf - T4 Short Training Run"""
import modal

app = modal.App("parameter-golf-t4-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "sentencepiece", "brotli", "zstandard", "numpy")
    .apt_install("git")
    .run_commands(
        "cd /root && git clone https://github.com/kailean/parameter-golf.git",
        "cd /root/parameter-golf && mkdir -p data/tokenizers data/datasets/fineweb10B_sp8192 data/datasets/fineweb10B_sp1024",
    )
)

vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/data": vol},
    timeout=1800,
    memory=16384,
)
def train_t4():
    import os, sys, io, subprocess
    import torch
    import brotli

    os.chdir("/root/parameter-golf")
    sys.path.insert(0, ".")

    # Symlink data from volume
    os.system("ln -sf /root/data/data/tokenizers/fineweb_8192_bpe.model /root/parameter-golf/data/tokenizers/fineweb_8192_bpe.model 2>/dev/null")
    os.system("ln -sf /root/data/data/tokenizers/fineweb_1024_bpe.model /root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model 2>/dev/null")
    os.system("ln -sf /root/data/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin /root/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin 2>/dev/null")
    os.system("ln -sf /root/data/data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin /root/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin 2>/dev/null")

    # SET ENV VARS FIRST before importing (so Hyperparameters picks them up)
    os.environ["VOCAB_SIZE"] = "8192"
    os.environ["NUM_LAYERS"] = "11"
    os.environ["MODEL_DIM"] = "512"
    os.environ["NUM_HEADS"] = "8"
    os.environ["NUM_KV_HEADS"] = "4"
    os.environ["MLP_MULT"] = "3"
    os.environ["TIE_EMBEDDINGS"] = "1"
    os.environ["DATA_PATH"] = "./data/datasets/fineweb10B_sp8192"
    os.environ["TOKENIZER_PATH"] = "./data/tokenizers/fineweb_8192_bpe.model"

    print("=" * 60)
    print("Parameter Golf - T4 Training Run (SP8192)")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Check data
    tok_path = "./data/tokenizers/fineweb_8192_bpe.model"
    val_path = "./data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
    train_path = "./data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin"
    for p in [tok_path, val_path, train_path]:
        print(f"  {p}: {'OK' if os.path.exists(p) else 'MISSING'} ({os.path.getsize(p)/1024/1024:.1f} MB if os.path.exists(p) else 0)")

    from train_gpt_kl import Hyperparameters, GPT, quantize_state_dict_int6, dequantize_state_dict_int6

    device = torch.device("cuda")
    args = Hyperparameters()
    print(f"Config: vocab={args.vocab_size}, layers={args.num_layers}, dim={args.model_dim}, mlp={args.mlp_mult}")

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, bigram_hash_size=args.bigram_hash_size,
        use_ortho_init=args.use_ortho_init, smear_enabled=args.smear_enabled,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims,
        ln_scale_enabled=args.ln_scale_enabled
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")

    # Compression test
    state_dict = model.state_dict()
    quantized = quantize_state_dict_int6(state_dict, use_gptq_lite=args.use_gptq_lite)
    buf = io.BytesIO()
    torch.save(quantized, buf)
    compressed = brotli.compress(buf.getvalue())
    print(f"Int6+Brotli: {len(compressed)/1024/1024:.2f} MB")

    # Run training (5 min, max 2000 iters)
    print("\n=== Starting T4 Training (5 min max) ===")
    os.environ["MAX_WALLCLOCK_SECONDS"] = "300"
    os.environ["ITERATIONS"] = "2000"
    os.environ["WORLD_SIZE"] = "1"

    result = subprocess.run(["python3", "train_gpt_kl.py"], capture_output=True, text=True, timeout=600)
    # Print last 3000 chars of stdout (training output)
    if result.stdout:
        lines = result.stdout.strip().split('\n')
        # Print last 50 lines
        for line in lines[-50:]:
            print(line)
    if result.stderr and result.returncode != 0:
        print("STDERR (last 20 lines):")
        for line in result.stderr.strip().split('\n')[-20:]:
            print(line)
    print(f"\nReturn code: {result.returncode}")

    vol.commit()
    print("\n=== Training complete ===")


@app.local_entrypoint()
def main():
    train_t4.remote()
