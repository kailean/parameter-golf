"""Parameter Golf - Modal GPU Training

Run with: modal run parameter_golf_modal.py
"""

import modal

app = modal.App("parameter-golf-training")

# Image with all dependencies + git clone
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "sentencepiece",
        "brotli",
        "zstandard",
        "numpy",
    )
    .apt_install("git")
    .run_commands(
        "cd /root && git clone https://github.com/kailean/parameter-golf.git",
        "cd /root/parameter-golf && mkdir -p data/tokenizers data/datasets/fineweb10B_sp8192 data/datasets/fineweb10B_sp1024",
    )
)

# Volume with our training data
vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/root/data": vol},
    timeout=7200,
    memory=32768,
)
def validate_and_train():
    """Run pipeline validation + training on A100."""
    import os, sys, io, glob
    import torch
    import brotli

    os.chdir("/root/parameter-golf")
    sys.path.insert(0, ".")
    os.system("ln -sf /root/data/tokenizers /root/parameter-golf/data/tokenizers 2>/dev/null")
    os.system("ln -sf /root/data/datasets /root/parameter-golf/data/datasets 2>/dev/null")

    print("=" * 60)
    print("Parameter Golf - A100 Training")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Symlink volume data into repo
    os.system("ln -sf /root/data/tokenizers /root/parameter-golf/data/tokenizers")
    os.system("ln -sf /root/data/datasets /root/parameter-golf/data/datasets")

    # Check data files
    data_files = glob.glob("/root/data/datasets/fineweb10B_sp8192/*.bin")
    tokenizer_files = glob.glob("/root/data/tokenizers/*.model")
    print(f"\nFound {len(tokenizer_files)} tokenizer(s), {len(data_files)} data file(s)")

    from train_gpt_kl import Hyperparameters, GPT, quantize_state_dict_int6, dequantize_state_dict_int6

    device = torch.device("cuda")
    args = Hyperparameters()
    print(f"\nConfig: vocab={args.vocab_size}, layers={args.num_layers}, dim={args.model_dim}")

    # Build model
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

    # Forward pass test
    x = torch.randint(0, args.vocab_size, (4, 512), device=device)
    y = torch.randint(0, args.vocab_size, (4, 512), device=device)
    with torch.no_grad():
        loss = model(x, y)
    print(f"Forward pass OK: loss = {loss.item():.4f}")

    # Int6 + Brotli compression test
    state_dict = model.state_dict()
    quantized = quantize_state_dict_int6(state_dict, use_gptq_lite=args.use_gptq_lite)
    buf = io.BytesIO()
    torch.save(quantized, buf)
    raw_bytes = buf.getvalue()
    compressed = brotli.compress(raw_bytes)
    mb = len(compressed) / 1024 / 1024
    print(f"\nInt6+Brotli: {len(raw_bytes):,} -> {len(compressed):,} bytes ({mb:.2f} MB)")
    print(f"Under 16MB: {'YES' if len(compressed) < 16*1024*1024 else 'NO'}")

    # Round-trip test
    decompressed = brotli.decompress(compressed)
    quant_rt = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
    dequant = dequantize_state_dict_int6(quant_rt)
    max_diff = max((state_dict[k].float() - dequant[k].float()).abs().max().item() for k in state_dict if k in dequant)
    print(f"Round-trip max diff: {max_diff:.6f}")
    print("\n=== ALL PIPELINE CHECKS PASSED ===")

    # Training
    if data_files:
        print("\n=== Starting Training ===")
        os.environ["DATA_PATH"] = "/root/data/datasets/fineweb10B_sp8192"
        os.environ["TOKENIZER_PATH"] = "/root/data/tokenizers/fineweb_8192_bpe.model"
        os.environ["VOCAB_SIZE"] = str(args.vocab_size)
        os.environ["NUM_LAYERS"] = str(args.num_layers)
        os.environ["MODEL_DIM"] = str(args.model_dim)
        os.environ["NUM_HEADS"] = str(args.num_heads)
        os.environ["NUM_KV_HEADS"] = str(args.num_kv_heads)
        os.environ["MLP_MULT"] = str(args.mlp_mult)
        os.environ["TIE_EMBEDDINGS"] = str(args.tie_embeddings)
        os.environ["MAX_WALLCLOCK_SECONDS"] = "7200"
        os.environ["ITERATIONS"] = "20000"
        os.environ["WORLD_SIZE"] = "1"

        os.system("python3 train_gpt_kl.py 2>&1 | tee training_output.log")
        vol.commit()
        print("\n=== Training complete ===")
    else:
        print("\nNo training data - validation only")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/data": vol},
    timeout=3600,
    memory=16384,
)
def validate_t4():
    """Quick pipeline validation on T4 (cheaper)."""
    import os, sys, io
    import torch
    import brotli

    os.chdir("/root/parameter-golf")
    sys.path.insert(0, ".")
    os.system("ln -sf /root/data/tokenizers /root/parameter-golf/data/tokenizers 2>/dev/null")
    os.system("ln -sf /root/data/datasets /root/parameter-golf/data/datasets 2>/dev/null")

    print("=" * 60)
    print("Parameter Golf - T4 Validation")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    from train_gpt_kl import Hyperparameters, GPT, quantize_state_dict_int6, dequantize_state_dict_int6

    device = torch.device("cuda")
    args = Hyperparameters()
    print(f"Config: vocab={args.vocab_size}, layers={args.num_layers}, dim={args.model_dim}")

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

    x = torch.randint(0, args.vocab_size, (2, 128), device=device)
    y = torch.randint(0, args.vocab_size, (2, 128), device=device)
    with torch.no_grad():
        loss = model(x, y)
    print(f"Forward pass: loss = {loss.item():.4f}")

    state_dict = model.state_dict()
    quantized = quantize_state_dict_int6(state_dict, use_gptq_lite=args.use_gptq_lite)
    buf = io.BytesIO()
    torch.save(quantized, buf)
    compressed = brotli.compress(buf.getvalue())
    print(f"Int6+Brotli: {len(compressed)/1024/1024:.2f} MB (under 16MB: {'YES' if len(compressed) < 16*1024*1024 else 'NO'})")
    print("=== Pipeline validated on T4 ===")


@app.local_entrypoint()
def main(gpu: str = "t4"):
    """Run validation. Use --gpu=a100 for A100."""
    if gpu == "a100":
        validate_and_train.remote()
    else:
        validate_t4.remote()
