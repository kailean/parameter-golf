#!/usr/bin/env python3
"""Test custom serializer + compression vs torch.save + byte shuffle."""
import sys, os, io, math
from collections import Counter

import torch
import numpy as np
import brotli
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt_kl import GPT, Hyperparameters, quantize_state_dict_int6
from custom_serializer import serialize_quantized, deserialize_quantized, roundtrip_test


def byte_entropy(data: bytes) -> tuple[float, int]:
    counter = Counter(data)
    total = len(data)
    entropy = -sum((c / total) * math.log2(c / total) for c in counter.values())
    unique = len(counter)
    return entropy, unique


def build_model(dim=640, mlp_mult=4):
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
    return model, args


def test_config(dim, mlp_mult, label):
    print(f"\n{'='*60}")
    print(f"  {label}: dim={dim}, mlp_mult={mlp_mult}")
    print(f"{'='*60}")
    
    model, _ = build_model(dim, mlp_mult)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    
    # Quantize
    quant_obj, stats = quantize_state_dict_int6(model.state_dict(), use_gptq_lite=True, embed_bits=8)
    payload = stats["int6_payload_bytes"]
    print(f"Payload: {payload:,} bytes ({payload/1e6:.2f}MB)")
    
    # Method 1: torch.save (old way)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    torchsave_raw = buf.getvalue()
    
    # Method 2: custom serializer
    custom_raw = serialize_quantized(quant_obj)
    
    print(f"\nRAW SIZES:")
    print(f"  torch.save:       {len(torchsave_raw):>12,} ({len(torchsave_raw)/1e6:.2f}MB)")
    print(f"  custom_serializer: {len(custom_raw):>12,} ({len(custom_raw)/1e6:.2f}MB)")
    print(f"  SAVINGS:           {len(torchsave_raw)-len(custom_raw):>12,} ({(len(torchsave_raw)-len(custom_raw))/1e6:.2f}MB) = {100*(len(torchsave_raw)-len(custom_raw))/len(torchsave_raw):.1f}%")
    
    # Roundtrip test
    print(f"\nROUNDTRIP TEST:")
    restored = deserialize_quantized(custom_raw)
    
    # Compare dequantized outputs
    from train_gpt_kl import dequantize_state_dict_int6
    sd_orig = dequantize_state_dict_int6(quant_obj)
    sd_rest = dequantize_state_dict_int6(restored)
    
    max_diff = 0.0
    all_match = True
    for name in sd_orig:
        if name not in sd_rest:
            print(f"  MISSING: {name}")
            all_match = False
            continue
        diff = (sd_orig[name].float() - sd_rest[name].float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 1e-6:
            print(f"  DIFF {name}: {diff:.8f}")
            all_match = False
    
    if all_match:
        print(f"  ✅ ALL MATCH (max_diff={max_diff:.2e})")
    else:
        print(f"  ❌ MISMATCH (max_diff={max_diff:.2e})")
    
    # Compression tests
    print(f"\nCOMPRESSION (brotli quality=11):")
    brotli_torchsave = brotli.compress(torchsave_raw, quality=11)
    brotli_custom = brotli.compress(custom_raw, quality=11)
    
    print(f"  brotli(torch.save):  {len(brotli_torchsave):>12,} ({len(brotli_torchsave)/1e6:.2f}MB)")
    print(f"  brotli(custom):      {len(brotli_custom):>12,} ({len(brotli_custom)/1e6:.2f}MB)")
    print(f"  SAVINGS:              {len(brotli_torchsave)-len(brotli_custom):>12,} ({(len(brotli_torchsave)-len(brotli_custom))/1e6:.2f}MB) = {100*(len(brotli_torchsave)-len(brotli_custom))/len(brotli_torchsave):.1f}%")
    
    # zstd comparison
    zstd_custom = zstandard.ZstdCompressor(level=22).compress(custom_raw)
    print(f"  zstd(custom):         {len(zstd_custom):>12,} ({len(zstd_custom)/1e6:.2f}MB)")
    
    # Entropy
    e_ts, u_ts = byte_entropy(torchsave_raw)
    e_cu, u_cu = byte_entropy(custom_raw)
    print(f"\nENTROPY:")
    print(f"  torch.save:  {e_ts:.2f} bits, {u_ts}/256 unique")
    print(f"  custom:      {e_cu:.2f} bits, {u_cu}/256 unique")
    
    # Code size
    code_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_gpt_kl.py")
    with open(code_path) as f:
        code = f.read()
    code_bytes = len(code.encode("utf-8"))
    
    # Also count custom_serializer as code
    cs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_serializer.py")
    with open(cs_path) as f:
        cs_code = f.read()
    cs_bytes = len(cs_code.encode("utf-8"))
    total_code = code_bytes + cs_bytes
    
    best_compressed = min(len(brotli_custom), len(zstd_custom))
    submission = best_compressed + total_code
    
    print(f"\nSUBMISSION ESTIMATE:")
    print(f"  Code:           {total_code:>12,} ({total_code/1e6:.2f}MB)")
    print(f"  Best model:     {best_compressed:>12,} ({best_compressed/1e6:.2f}MB)")
    print(f"  TOTAL:          {submission:>12,} ({submission/1e6:.2f}MB)")
    print(f"  TARGET:          16,000,000 (16.00MB)")
    print(f"  FITS?            {'YES ✅' if submission < 16_000_000 else 'NO ❌'}")
    print(f"  HEADROOM:        {(16_000_000 - submission)/1e6:.2f}MB" if submission < 16_000_000 else f"  OVER BY:        {(submission - 16_000_000)/1e6:.2f}MB")
    
    return {
        "dim": dim, "mlp_mult": mlp_mult, "params": total_params,
        "payload": payload, "custom_raw": len(custom_raw),
        "brotli_custom": len(brotli_custom), "zstd_custom": len(zstd_custom),
        "submission": submission, "fits": submission < 16_000_000,
    }


if __name__ == "__main__":
    results = []
    
    # Test 640d/4x (current config — but over 16MB!)
    # r = test_config(640, 4, "640d MLP×4 (170M params, likely over)")
    # results.append(r)
    
    # Test 640d/3x
    r = test_config(640, 3, "640d MLP×3 (~45M params)")
    results.append(r)
    
    # Test 768d/4x  
    r = test_config(768, 4, "768d MLP×4 (~95M params)")
    results.append(r)
    
    # Test 768d/3x
    r = test_config(768, 3, "768d MLP×3 (~74M params)")
    results.append(r)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Params':>10} {'Brotli':>10} {'Submission':>12} {'Fits':>6}")
    print(f"{'-'*65}")
    for r in results:
        print(f"dim={r['dim']} mlp={r['mlp_mult']}x{' '*12} {r['params']:>10,} {r['brotli_custom']/1e6:>8.2f}MB {r['submission']/1e6:>10.2f}MB {'✅' if r['fits'] else '❌':>6}")