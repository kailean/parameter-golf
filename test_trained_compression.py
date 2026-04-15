#!/usr/bin/env python3
"""Test compression ratio on SIMULATED trained weights.
Trained weights have very different entropy distribution from random/untrained:
  - Weight matrices become low-rank (singular value decay)
  - int6 quantized trained data has clustered values (not uniform)
  - SOTA ratio: 47M params → 15.99MB = 0.34 bytes/param

We simulate this by:
1. Creating a model with trained-like weight distribution (low-rank + noise)
2. Quantizing to int6 with GPTQ-lite
3. Compressing with our custom serializer + brotli
"""
import sys, os, io
import torch
import numpy as np
import brotli
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt_kl import GPT, Hyperparameters, quantize_state_dict_int6
from custom_serializer import serialize_quantized


def make_trained_like_weights(model, rank_frac=0.3, noise_std=0.02):
    """Replace random init with trained-like weights: low-rank + structured noise."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim < 2:
                continue
            # Create low-rank structure (trained weights are approximately low-rank)
            m, n = p.shape
            rank = max(1, int(min(m, n) * rank_frac))
            U = torch.randn(m, rank) * (1.0 / rank**0.5)
            V = torch.randn(rank, n) * (1.0 / rank**0.5)
            low_rank = U @ V
            
            # Scale to match typical trained weight magnitudes
            # Trained transformer weights tend to have std ~0.01-0.1
            target_std = 0.05 if "emb" in name else 0.02
            low_rank = low_rank * (target_std / max(low_rank.std(), 1e-8))
            
            # Add noise (breaks perfect low-rank, like real trained weights)
            noise = torch.randn_like(p) * noise_std * target_std
            p.copy_(low_rank + noise)


def test_config(dim, mlp_mult, label, trained=True):
    print(f"\n{'='*60}")
    print(f"  {label}: dim={dim}, mlp_mult={mlp_mult}, {'TRAINED-LIKE' if trained else 'UNTRAINED'}")
    print(f"{'='*60}")
    
    args = Hyperparameters()
    args.vocab_size = 8192
    args.num_layers = 11
    args.model_dim = dim
    args.num_heads = 10
    args.num_kv_heads = 5
    args.mlp_mult = mlp_mult
    args.tie_embeddings = True
    args.tied_embed_init_std = 0.005
    args.logit_softcap = 30.0
    args.rope_base = 10000.0
    args.qk_gain_init = 5.25
    args.bigram_hash_size = 0
    args.use_ortho_init = True
    args.smear_enabled = True
    args.xsa_last_n = 4
    args.rope_dims = 16
    args.ln_scale_enabled = True
    args.depth_recurrence = True
    args.recurrence_layers = [3, 4, 5]
    args.recurrence_loops = 2
    args.parallel_residuals = True
    args.parallel_res_start = 7
    args.leaky_relu_slope = 0.5
    args.embed_bits = 8
    args.use_gptq_lite = True
    
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_hash_size=args.bigram_hash_size, use_ortho_init=args.use_ortho_init,
        smear_enabled=args.smear_enabled, xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale_enabled=args.ln_scale_enabled,
        depth_recurrence=args.depth_recurrence,
        recurrence_layers=args.recurrence_layers,
        recurrence_loops=args.recurrence_loops,
        parallel_residuals=args.parallel_residuals,
        parallel_res_start=args.parallel_res_start,
        leaky_slope=args.leaky_relu_slope,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    
    if trained:
        make_trained_like_weights(model)
    
    # Quantize
    quant_obj, stats = quantize_state_dict_int6(model.state_dict(), use_gptq_lite=True, embed_bits=8)
    payload = stats["int6_payload_bytes"]
    print(f"Payload: {payload:,} bytes ({payload/1e6:.2f}MB)")
    
    # Custom serializer
    custom_raw = serialize_quantized(quant_obj)
    print(f"Custom raw: {len(custom_raw):,} bytes ({len(custom_raw)/1e6:.2f}MB)")
    
    # Compression
    brotli_custom = brotli.compress(custom_raw, quality=11)
    zstd_custom = zstandard.ZstdCompressor(level=22).compress(custom_raw)
    
    print(f"brotli:    {len(brotli_custom):>12,} ({len(brotli_custom)/1e6:.2f}MB)")
    print(f"zstd:      {len(zstd_custom):>12,} ({len(zstd_custom)/1e6:.2f}MB)")
    
    # Ratio
    ratio = len(brotli_custom) / total_params
    print(f"brotli ratio: {ratio:.4f} bytes/param")
    print(f"SOTA ratio:   0.34 bytes/param (PR #1530)")
    
    # Submission
    code_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_gpt_kl.py")
    cs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_serializer.py")
    code_bytes = len(open(code_path).read().encode()) + len(open(cs_path).read().encode())
    best = min(len(brotli_custom), len(zstd_custom))
    submission = best + code_bytes
    
    print(f"\nSubmission: {submission:,} bytes ({submission/1e6:.2f}MB)")
    print(f"Fits 16MB? {'YES ✅' if submission < 16_000_000 else 'NO ❌'}")
    if submission < 16_000_000:
        print(f"Headroom: {(16_000_000 - submission)/1e6:.2f}MB")
    
    # Per-tensor entropy analysis
    from collections import Counter
    import math
    
    quantized = quant_obj.get("quantized", {})
    print(f"\nPer-tensor compression (top 5):")
    tensor_info = []
    for name in sorted(quantized.keys()):
        v = quantized[name]
        if isinstance(v, np.ndarray):
            raw = v.tobytes()
            compressed = len(brotli.compress(raw, quality=11))
            ratio_t = compressed / max(len(raw), 1)
            # Entropy
            counter = Counter(raw)
            total = len(raw)
            ent = -sum((c/total) * math.log2(c/total) for c in counter.values())
            tensor_info.append((name, len(raw), compressed, ratio_t, ent))
    
    tensor_info.sort(key=lambda x: -x[1])
    for name, raw_s, comp_s, ratio_t, ent in tensor_info[:5]:
        print(f"  {name}: raw={raw_s/1e6:.2f}MB brotli={comp_s/1e6:.2f}MB ratio={ratio_t:.3f} entropy={ent:.2f}")
    
    return {
        "label": label, "params": total_params, "payload": payload,
        "brotli": len(brotli_custom), "submission": submission,
        "ratio": len(brotli_custom) / total_params,
        "fits": submission < 16_000_000,
    }


if __name__ == "__main__":
    results = []
    
    # Test trained-like weights at multiple configs
    for dim, mlp, label in [
        (640, 3, "640d MLP×3"),
        (768, 4, "768d MLP×4"),
        (768, 3, "768d MLP×3"),
    ]:
        r = test_config(dim, mlp, label, trained=True)
        results.append(r)
    
    print(f"\n{'='*70}")
    print(f"  TRAINED-LIKE COMPRESSION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Params':>10} {'Brotli':>10} {'Ratio':>8} {'Submiss':>10} {'Fits':>6}")
    print(f"{'-'*66}")
    for r in results:
        print(f"{r['label']:<20} {r['params']:>10,} {r['brotli']/1e6:>8.2f}MB {r['ratio']:>7.4f} {r['submission']/1e6:>8.2f}MB {'✅' if r['fits'] else '❌':>6}")