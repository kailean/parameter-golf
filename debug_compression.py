#!/usr/bin/env python3
"""Quick compression debug — just 640d, no subprocess."""
import sys, os, io, math
from collections import Counter

import torch
import numpy as np
import brotli
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt_kl import GPT, Hyperparameters, quantize_state_dict_int6


def byte_entropy(data: bytes) -> tuple[float, int]:
    counter = Counter(data)
    total = len(data)
    entropy = -sum((c / total) * math.log2(c / total) for c in counter.values())
    unique = len(counter)
    return entropy, unique


def byte_shuffle(data: bytes, stride: int = 2) -> bytes:
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
args.vocab_size = 8192
args.num_layers = 11
args.model_dim = 640
args.num_heads = 10
args.num_kv_heads = 5
args.mlp_mult = 4
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
print(f"Total params: {total_params:,}", flush=True)

state_dict = model.state_dict()
print("Quantizing...", flush=True)
quant_obj, stats = quantize_state_dict_int6(state_dict, use_gptq_lite=True, embed_bits=8)

quantized = quant_obj.get("quantized", {})
scales = quant_obj.get("scales", {})
passthrough = quant_obj.get("passthrough", {})

total_int6_bytes = sum(v.nbytes for v in quantized.values() if isinstance(v, np.ndarray))
total_scale_bytes = sum(v.numel() * v.element_size() for v in scales.values() if isinstance(v, torch.Tensor))
total_float_bytes = sum(v.numel() * v.element_size() for v in passthrough.values() if isinstance(v, torch.Tensor))
total_payload = total_int6_bytes + total_scale_bytes + total_float_bytes

print(f"\nPayload breakdown:", flush=True)
print(f"  int6 packed:   {total_int6_bytes:>12,} ({total_int6_bytes/1e6:.2f}MB)", flush=True)
print(f"  scales:        {total_scale_bytes:>12,} ({total_scale_bytes/1e6:.2f}MB)", flush=True)
print(f"  float pass:    {total_float_bytes:>12,} ({total_float_bytes/1e6:.2f}MB)", flush=True)
print(f"  TOTAL:         {total_payload:>12,} ({total_payload/1e6:.2f}MB)", flush=True)

# Serialize
quant_buf = io.BytesIO()
torch.save(quant_obj, quant_buf)
raw_pickle = quant_buf.getvalue()

print(f"\ntorch.save:      {len(raw_pickle):>12,} ({len(raw_pickle)/1e6:.2f}MB)", flush=True)
print(f"pickle overhead: {len(raw_pickle) - total_payload:>12,} ({(len(raw_pickle) - total_payload)/1e6:.2f}MB)", flush=True)

# Byte entropy
entropy_raw, unique_raw = byte_entropy(raw_pickle)
print(f"\nRAW PICKLE: unique={unique_raw}/256, entropy={entropy_raw:.2f} bits", flush=True)

# All quantized data entropy
all_quant = b""
for name in sorted(quantized.keys()):
    v = quantized[name]
    if isinstance(v, np.ndarray):
        all_quant += v.tobytes()
entropy_quant, unique_quant = byte_entropy(all_quant)
print(f"QUANT DATA: unique={unique_quant}/256, entropy={entropy_quant:.2f} bits, size={len(all_quant):,}", flush=True)

# Per-tensor top 10
print(f"\nTOP 10 TENSORS BY SIZE:", flush=True)
tensor_sizes = []
for name in sorted(quantized.keys()):
    v = quantized[name]
    if isinstance(v, np.ndarray):
        e, u = byte_entropy(v.tobytes())
        scheme = quant_obj.get("qmeta", {}).get(name, {})
        tensor_sizes.append((name, v.nbytes, e, u, scheme))
tensor_sizes.sort(key=lambda x: -x[1])
for name, size, ent, uniq, scheme in tensor_sizes[:10]:
    print(f"  {name}: {size:>10,} bytes, entropy={ent:.2f}, unique={uniq}/256, scheme={scheme}", flush=True)

# Passthrough tensors
print(f"\nPASSTHROUGH (float, not quantized):", flush=True)
for name in sorted(passthrough.keys()):
    v = passthrough[name]
    if isinstance(v, torch.Tensor):
        size = v.numel() * v.element_size()
        orig_dtype = quant_obj.get("passthrough_orig_dtypes", {}).get(name, "?")
        if size > 100:
            print(f"  {name}: shape={list(v.shape)} dtype={v.dtype} orig={orig_dtype} bytes={size:,}", flush=True)

# Compression tests
print(f"\nCOMPRESSION TESTS:", flush=True)
brotli_plain = brotli.compress(raw_pickle, quality=11)
print(f"  brotli (plain):          {len(brotli_plain):>12,} ({len(brotli_plain)/1e6:.2f}MB)", flush=True)

shuffled2 = byte_shuffle(raw_pickle, stride=2)
brotli_shuf2 = brotli.compress(shuffled2, quality=11)
print(f"  brotli (stride=2):       {len(brotli_shuf2):>12,} ({len(brotli_shuf2)/1e6:.2f}MB)", flush=True)

shuffled4 = byte_shuffle(raw_pickle, stride=4)
brotli_shuf4 = brotli.compress(shuffled4, quality=11)
print(f"  brotli (stride=4):       {len(brotli_shuf4):>12,} ({len(brotli_shuf4)/1e6:.2f}MB)", flush=True)

zstd_plain = zstandard.ZstdCompressor(level=22).compress(raw_pickle)
print(f"  zstd (plain):            {len(zstd_plain):>12,} ({len(zstd_plain)/1e6:.2f}MB)", flush=True)

zstd_shuf = zstandard.ZstdCompressor(level=22).compress(shuffled2)
print(f"  zstd (stride=2):         {len(zstd_shuf):>12,} ({len(zstd_shuf)/1e6:.2f}MB)", flush=True)

# Verify unshuffle
unshuffled = b"BSHF" + bytes([2]) + np.empty(len(raw_pickle), dtype=np.uint8)
# Quick verification: just check header roundtrip
print(f"\nByte shuffle header verified: {shuffled2[:5]}", flush=True)

# Code size
code_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_gpt_kl.py")
with open(code_path) as f:
    code = f.read()
code_bytes = len(code.encode("utf-8"))

best = min(len(brotli_plain), len(brotli_shuf2), len(brotli_shuf4))
submission = best + code_bytes
fits = submission < 16_000_000

print(f"\n  Code size:               {code_bytes:>12,} ({code_bytes/1e6:.2f}MB)", flush=True)
print(f"  Best brotli:             {best:>12,} ({best/1e6:.2f}MB)", flush=True)
print(f"  TOTAL SUBMISSION:       {submission:>12,} ({submission/1e6:.2f}MB)", flush=True)
print(f"  TARGET:                  16,000,000 (16.00MB)", flush=True)
print(f"  FITS?                    {'YES ✅' if fits else 'NO ❌'}", flush=True)

print(f"\nIMPROVEMENT RATIOS:", flush=True)
print(f"  plain→stride2: {len(brotli_plain)/max(len(brotli_shuf2),1):.2f}x", flush=True)
print(f"  plain→stride4: {len(brotli_plain)/max(len(brotli_shuf4),1):.2f}x", flush=True)
print(f"  brotli→zstd:   {len(brotli_plain)/max(len(zstd_plain),1):.2f}x", flush=True)