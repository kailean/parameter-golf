#!/usr/bin/env python3
"""Run compression debug directly on Modal (no subprocess)."""
import modal
import os
import sys

app = modal.App("compression-debug-v2")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.11")
    .pip_install("torch==2.6.0", "numpy", "sentencepiece", "brotli", "zstandard")
    .add_local_file(
        "/Users/kaileanhard/.openclaw/workspace/parameter-golf/train_gpt_kl.py",
        "/root/train_gpt_kl.py",
        copy=False,
    )
)

@app.function(image=image, gpu="H100", timeout=600)
def debug():
    import torch
    import numpy as np
    import brotli
    import zstandard
    import io
    import math
    from collections import Counter

    sys.path.insert(0, "/root")
    from train_gpt_kl import GPT, Hyperparameters, quantize_state_dict_int6

    def byte_entropy(data):
        counter = Counter(data)
        total = len(data)
        entropy = -sum((c / total) * math.log2(c / total) for c in counter.values())
        unique = len(counter)
        return entropy, unique

    def byte_shuffle(data, stride=2):
        src = np.frombuffer(data, dtype=np.uint8)
        n = len(src)
        out = np.empty(n, dtype=np.uint8)
        offset = 0
        for pos in range(stride):
            chunk = src[pos::stride]
            out[offset:offset + len(chunk)] = chunk
            offset += len(chunk)
        return b"BSHF" + bytes([stride]) + out.tobytes()

    # Build 640d model
    print("Building 640d model...", flush=True)
    args = Hyperparameters()
    args.vocab_size = 8192; args.num_layers = 11; args.model_dim = 640
    args.num_heads = 10; args.num_kv_heads = 5; args.mlp_mult = 4
    args.tie_embeddings = True; args.tied_embed_init_std = 0.005
    args.logit_softcap = 30.0; args.rope_base = 10000.0; args.qk_gain_init = 5.25
    args.bigram_hash_size = 0; args.use_ortho_init = True
    args.smear_enabled = True; args.xsa_last_n = 4; args.rope_dims = 16
    args.ln_scale_enabled = True; args.depth_recurrence = True
    args.recurrence_layers = [3, 4, 5]; args.recurrence_loops = 2
    args.parallel_residuals = True; args.parallel_res_start = 7
    args.leaky_relu_slope = 0.5; args.embed_bits = 8; args.use_gptq_lite = True

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_hash_size=args.bigram_hash_size, use_ortho_init=args.use_ortho_init,
        smear_enabled=args.smear_enabled, xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale_enabled=args.ln_scale_enabled,
        depth_recurrence=args.depth_recurrence, recurrence_layers=args.recurrence_layers,
        recurrence_loops=args.recurrence_loops, parallel_residuals=args.parallel_residuals,
        parallel_res_start=args.parallel_res_start, leaky_slope=args.leaky_relu_slope,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}", flush=True)

    state_dict = model.state_dict()
    print("Quantizing...", flush=True)
    quant_obj, stats = quantize_state_dict_int6(state_dict, use_gptq_lite=True, embed_bits=8)

    quantized = quant_obj.get("quantized", {})
    scales = quant_obj.get("scales", {})
    passthrough = quant_obj.get("passthrough", {})

    total_int6 = sum(v.nbytes for v in quantized.values() if isinstance(v, np.ndarray))
    total_scales = sum(v.numel() * v.element_size() for v in scales.values() if isinstance(v, torch.Tensor))
    total_float = sum(v.numel() * v.element_size() for v in passthrough.values() if isinstance(v, torch.Tensor))
    total_payload = total_int6 + total_scales + total_float

    print(f"\nPayload: int6={total_int6/1e6:.2f}MB scales={total_scales/1e6:.2f}MB float={total_float/1e6:.2f}MB TOTAL={total_payload/1e6:.2f}MB", flush=True)

    # Serialize
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    raw_pickle = quant_buf.getvalue()

    print(f"torch.save: {len(raw_pickle)/1e6:.2f}MB (overhead: {(len(raw_pickle)-total_payload)/1e6:.2f}MB)", flush=True)

    # Entropy diagnostics
    e_raw, u_raw = byte_entropy(raw_pickle)
    print(f"\nRAW PICKLE: unique={u_raw}/256 entropy={e_raw:.2f}bits", flush=True)

    all_quant = b"".join(v.tobytes() for v in quantized.values() if isinstance(v, np.ndarray))
    e_q, u_q = byte_entropy(all_quant)
    print(f"QUANT DATA: unique={u_q}/256 entropy={e_q:.2f}bits size={len(all_quant)/1e6:.2f}MB", flush=True)

    # Top 10 tensors
    print(f"\nTOP 10 TENSORS:", flush=True)
    ts = [(n, v.nbytes, *byte_entropy(v.tobytes()), quant_obj.get("qmeta",{}).get(n,{}))
          for n, v in quantized.items() if isinstance(v, np.ndarray)]
    ts.sort(key=lambda x: -x[1])
    for n, sz, e, u, sc in ts[:10]:
        print(f"  {n}: {sz:,}B entropy={e:.2f} unique={u}/256 scheme={sc}", flush=True)

    # Passthrough
    print(f"\nPASSTHROUGH (>100B):", flush=True)
    for n, v in sorted(passthrough.items()):
        if isinstance(v, torch.Tensor) and v.numel() * v.element_size() > 100:
            sz = v.numel() * v.element_size()
            od = quant_obj.get("passthrough_orig_dtypes", {}).get(n, "?")
            print(f"  {n}: shape={list(v.shape)} dtype={v.dtype} orig={od} bytes={sz:,}", flush=True)

    # Compression
    print(f"\nCOMPRESSION:", flush=True)
    b_plain = brotli.compress(raw_pickle, quality=11)
    print(f"  brotli plain:    {len(b_plain)/1e6:.2f}MB", flush=True)

    b_shuf2 = brotli.compress(byte_shuffle(raw_pickle, 2), quality=11)
    print(f"  brotli shuf2:    {len(b_shuf2)/1e6:.2f}MB ({len(b_plain)/len(b_shuf2):.2f}x better)", flush=True)

    b_shuf4 = brotli.compress(byte_shuffle(raw_pickle, 4), quality=11)
    print(f"  brotli shuf4:    {len(b_shuf4)/1e6:.2f}MB ({len(b_plain)/len(b_shuf4):.2f}x better)", flush=True)

    z_plain = zstandard.ZstdCompressor(level=22).compress(raw_pickle)
    print(f"  zstd plain:      {len(z_plain)/1e6:.2f}MB", flush=True)

    z_shuf2 = zstandard.ZstdCompressor(level=22).compress(byte_shuffle(raw_pickle, 2))
    print(f"  zstd shuf2:      {len(z_shuf2)/1e6:.2f}MB ({len(z_plain)/len(z_shuf2):.2f}x better)", flush=True)

    # Code + submission
    with open("/root/train_gpt_kl.py") as f:
        code = f.read()
    code_bytes = len(code.encode("utf-8"))
    best = min(len(b_plain), len(b_shuf2), len(b_shuf4))
    submission = best + code_bytes
    fits = submission < 16_000_000

    print(f"\nCode: {code_bytes:,}B ({code_bytes/1e6:.2f}MB)", flush=True)
    print(f"Best brotli: {best/1e6:.2f}MB", flush=True)
    print(f"SUBMISSION: {submission/1e6:.2f}MB {'FITS ✅' if fits else 'OVER ❌'} (limit: 16.00MB)", flush=True)

    # Also test 768d
    print(f"\n{'='*60}", flush=True)
    print("Building 768d model...", flush=True)
    args2 = Hyperparameters()
    args2.vocab_size = 8192; args2.num_layers = 11; args2.model_dim = 768
    args2.num_heads = 12; args2.num_kv_heads = 6; args2.mlp_mult = 4
    args2.tie_embeddings = True; args2.tied_embed_init_std = 0.005
    args2.logit_softcap = 30.0; args2.rope_base = 10000.0; args2.qk_gain_init = 5.25
    args2.bigram_hash_size = 0; args2.use_ortho_init = True
    args2.smear_enabled = True; args2.xsa_last_n = 4; args2.rope_dims = 16
    args2.ln_scale_enabled = True; args2.depth_recurrence = True
    args2.recurrence_layers = [3, 4, 5]; args2.recurrence_loops = 2
    args2.parallel_residuals = True; args2.parallel_res_start = 7
    args2.leaky_relu_slope = 0.5; args2.embed_bits = 8; args2.use_gptq_lite = True

    model2 = GPT(
        vocab_size=args2.vocab_size, num_layers=args2.num_layers, model_dim=args2.model_dim,
        num_heads=args2.num_heads, num_kv_heads=args2.num_kv_heads, mlp_mult=args2.mlp_mult,
        tie_embeddings=args2.tie_embeddings, tied_embed_init_std=args2.tied_embed_init_std,
        logit_softcap=args2.logit_softcap, rope_base=args2.rope_base, qk_gain_init=args2.qk_gain_init,
        bigram_hash_size=args2.bigram_hash_size, use_ortho_init=args2.use_ortho_init,
        smear_enabled=args2.smear_enabled, xsa_last_n=args2.xsa_last_n,
        rope_dims=args2.rope_dims, ln_scale_enabled=args2.ln_scale_enabled,
        depth_recurrence=args2.depth_recurrence, recurrence_layers=args2.recurrence_layers,
        recurrence_loops=args2.recurrence_loops, parallel_residuals=args2.parallel_residuals,
        parallel_res_start=args2.parallel_res_start, leaky_slope=args2.leaky_relu_slope,
    )

    tp2 = sum(p.numel() for p in model2.parameters())
    print(f"Total params: {tp2:,}", flush=True)
    sd2 = model2.state_dict()
    qo2, _ = quantize_state_dict_int6(sd2, use_gptq_lite=True, embed_bits=8)

    qb2 = io.BytesIO()
    torch.save(qo2, qb2)
    rp2 = qb2.getvalue()

    e2, u2 = byte_entropy(rp2)
    print(f"RAW PICKLE: unique={u2}/256 entropy={e2:.2f}bits", flush=True)

    b2_plain = brotli.compress(rp2, quality=11)
    b2_shuf2 = brotli.compress(byte_shuffle(rp2, 2), quality=11)
    b2_shuf4 = brotli.compress(byte_shuffle(rp2, 4), quality=11)

    print(f"  brotli plain:    {len(b2_plain)/1e6:.2f}MB", flush=True)
    print(f"  brotli shuf2:    {len(b2_shuf2)/1e6:.2f}MB ({len(b2_plain)/len(b2_shuf2):.2f}x)", flush=True)
    print(f"  brotli shuf4:    {len(b2_shuf4)/1e6:.2f}MB ({len(b2_plain)/len(b2_shuf4):.2f}x)", flush=True)

    best2 = min(len(b2_plain), len(b2_shuf2), len(b2_shuf4))
    sub2 = best2 + code_bytes
    fits2 = sub2 < 16_000_000
    print(f"\n768d SUBMISSION: {sub2/1e6:.2f}MB {'FITS ✅' if fits2 else 'OVER ❌'}", flush=True)

    return "done"


@app.local_entrypoint()
def main():
    print("Running compression debug on H100...")
    result = debug.remote()
    print(f"Result: {result}")