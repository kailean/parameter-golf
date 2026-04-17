#!/usr/bin/env python3
"""
End-to-end GPTQ compression test on TRAINED model.
THE ONLY NUMBER THAT MATTERS: compressed size + post-quantization BPB.
"""
import sys, os, io, time, math
from collections import Counter
import torch, numpy as np, brotli, zstandard, sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt_kl import GPT, Hyperparameters, quantize_state_dict_int6, dequantize_state_dict_int6
from custom_serializer import serialize_quantized
from gptq_calibration import gptq_quantize_weight

CHECKPOINT = "/Volumes/MacStorageExtended/parameter-golf-data/checkpoints/final_model_2000step.pt"
TOKENIZER = "/Volumes/MacStorageExtended/parameter-golf-data/sp8192_cache/datasets/tokenizers/fineweb_8192_bpe.model"
VAL_DATA = "/Volumes/MacStorageExtended/parameter-golf-data/sp8192_cache/datasets/datasets/fineweb10B_sp8192/fineweb_val_0_of_1.bin"

def load_val_tokens():
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(VAL_DATA, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(VAL_DATA, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def eval_bpb(model, val_tokens, sp, seq_len=2048, max_batches=20):
    """Quick BPB eval on validation data."""
    device = next(model.parameters()).device
    model.eval()
    usable = (val_tokens.numel() - 1) // seq_len * seq_len
    tokens = val_tokens[:usable + 1].to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, min(usable, max_batches * seq_len), seq_len):
            x = tokens[i:i+seq_len].unsqueeze(0)
            y = tokens[i+1:i+seq_len+1].unsqueeze(0)
            loss = model(x, y)
            total_loss += loss.item() * seq_len
            total_tokens += seq_len
    
    val_loss = total_loss / total_tokens
    # BPB conversion
    val_bpb = val_loss / math.log(2) * 4.75  # approximate bytes/token for SP8192
    return val_loss, val_bpb

def build_model():
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
    return GPT(
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

def gptq_quantize_sd(sd, matrix_clip_sigmas=12.85, embed_clip_sigmas=20.0, matrix_bits=6, embed_bits=8):
    """Full GPTQ quantization with identity Hessian (no calibration data)."""
    _EMBED_PATTERNS = ("tok_emb.weight", "lm_head.weight")
    quantized = {}
    scales_dict = {}
    shapes = {}
    dtypes = {}
    passthrough = {}
    qmeta = {}
    passthrough_orig_dtypes = {}
    
    total_params = 0
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536 or t.ndim < 2:
            passthrough[name] = t
            continue
        
        total_params += t.numel()
        is_embed = any(p in name for p in _EMBED_PATTERNS)
        clip_sigmas = embed_clip_sigmas if is_embed else matrix_clip_sigmas
        clip_range = (2 ** (embed_bits - 1) - 1) if is_embed else (2 ** (matrix_bits - 1) - 1)
        bits = embed_bits if is_embed else matrix_bits
        
        # Identity Hessian (no calibration — this is the conservative baseline)
        H = torch.eye(t.shape[1])
        
        q, s = gptq_quantize_weight(t, H, clip_sigmas=clip_sigmas, clip_range=clip_range)
        
        # Store in our format
        q_np = q.numpy().tobytes()
        packed_np = np.frombuffer(q_np, dtype=np.uint8).copy()
        quantized[name] = packed_np
        scales_dict[name] = s
        shapes[name] = t.shape
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        
        if is_embed:
            qmeta[name] = {"scheme": "per_row_int8", "axis": 0}
        else:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
    
    quant_obj = {
        "__quant_format__": "gptq_mixed_v1",
        "quantized": quantized, "scales": scales_dict, "shapes": shapes,
        "dtypes": dtypes, "passthrough": passthrough,
        "qmeta": qmeta, "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    return quant_obj

def dequantize_gptq_sd(quant_obj):
    """Dequantize GPTQ format back to float.
    GPTQ stores quantized values as raw int8 bytes (1 byte per value, NOT int6 packed).
    """
    out = {}
    qmeta = quant_obj.get("qmeta", {})
    pt_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    shapes = quant_obj.get("shapes", {})
    
    for name, packed in quant_obj["quantized"].items():
        orig_shape = shapes[name]
        meta = qmeta.get(name, {})
        
        # GPTQ output is raw int8 (1 byte per value), reshape directly
        q_int8 = packed.reshape(orig_shape).astype(np.float32)
        q_t = torch.from_numpy(q_int8)
        
        s = quant_obj["scales"][name]
        if isinstance(s, torch.Tensor):
            s32 = s.to(dtype=torch.float32)
        else:
            s32 = torch.tensor(float(s), dtype=torch.float32)
        
        if meta.get("scheme") == "per_row" or s32.ndim > 0:
            out[name] = (q_t * s32.view(q_t.shape[0], *([1] * (q_t.ndim - 1)))).to(dtype=torch.bfloat16).contiguous()
        else:
            out[name] = (q_t * float(s32.item())).to(dtype=torch.bfloat16).contiguous()
    
    for name, t in quant_obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous() if isinstance(t, torch.Tensor) else torch.from_numpy(np.array(t))
        orig = pt_dtypes.get(name)
        out[name] = out_t.to(dtype=getattr(torch, orig)) if isinstance(orig, str) else out_t
    
    return out

if __name__ == "__main__":
    device = torch.device("cpu")
    
    print("Loading trained checkpoint...", flush=True)
    model = build_model()
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {total_params:,} params", flush=True)
    
    # === BASELINE: Our existing GPTQ-lite pipeline ===
    print("\n=== BASELINE: GPTQ-lite + custom serializer ===", flush=True)
    t0 = time.perf_counter()
    quant_lite, stats_lite = quantize_state_dict_int6(sd, use_gptq_lite=True, embed_bits=8)
    raw_lite = serialize_quantized(quant_lite)
    b_lite = brotli.compress(raw_lite, quality=11)
    t1 = time.perf_counter()
    
    # Dequantize and eval
    deq_lite = dequantize_state_dict_int6(quant_lite)
    model.load_state_dict(deq_lite)
    print(f"  Compressed: {len(b_lite):,} ({len(b_lite)/1e6:.2f}MB) [{t1-t0:.1f}s]", flush=True)
    
    # === TEST: Full GPTQ with identity Hessian ===
    print("\n=== TEST: Full GPTQ (identity Hessian) clip_sigmas=12.85 ===", flush=True)
    t0 = time.perf_counter()
    quant_gptq = gptq_quantize_sd(sd, matrix_clip_sigmas=12.85, embed_clip_sigmas=20.0)
    raw_gptq = serialize_quantized(quant_gptq)
    b_gptq = brotli.compress(raw_gptq, quality=11)
    t1 = time.perf_counter()
    
    deq_gptq = dequantize_gptq_sd(quant_gptq)
    model.load_state_dict(deq_gptq)
    print(f"  Compressed: {len(b_gptq):,} ({len(b_gptq)/1e6:.2f}MB) [{t1-t0:.1f}s]", flush=True)
    
    # === Compare per-tensor compression ===
    print("\n=== PER-TENSOR COMPRESSION COMPARISON ===", flush=True)
    for name in sorted(quant_lite["quantized"].keys())[:5]:
        lite_data = quant_lite["quantized"][name]
        gptq_data = quant_gptq["quantized"][name]
        if isinstance(lite_data, np.ndarray) and isinstance(gptq_data, np.ndarray):
            lb = len(brotli.compress(lite_data.tobytes(), quality=6))
            gb = len(brotli.compress(gptq_data.tobytes(), quality=6))
            print(f"  {name}: lite={lb:,} gptq={gb:,} diff={lb-gb:,} ({100*(lb-gb)/lb:.1f}%)", flush=True)
    
    # === THE NUMBER: total compressed size ===
    code_bytes = len(open("train_gpt_kl.py").read().encode()) + len(open("custom_serializer.py").read().encode())
    
    print(f"\n{'='*60}", flush=True)
    print(f"  THE NUMBER", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  GPTQ-lite:  {len(b_lite):,} bytes ({len(b_lite)/1e6:.2f}MB) + code {code_bytes:,} = {len(b_lite)+code_bytes:,} ({(len(b_lite)+code_bytes)/1e6:.2f}MB)", flush=True)
    print(f"  GPTQ-full:  {len(b_gptq):,} bytes ({len(b_gptq)/1e6:.2f}MB) + code {code_bytes:,} = {len(b_gptq)+code_bytes:,} ({(len(b_gptq)+code_bytes)/1e6:.2f}MB)", flush=True)
    print(f"  Savings:    {len(b_lite)-len(b_gptq):,} bytes ({(len(b_lite)-len(b_gptq))/1e6:.2f}MB) = {100*(len(b_lite)-len(b_gptq))/len(b_lite):.1f}%", flush=True)
    print(f"  Target:     16,000,000 bytes (16.00MB)", flush=True)
    print(f"  Fits?       GPTQ-lite: {'YES' if len(b_lite)+code_bytes < 16_000_000 else 'NO'} | GPTQ-full: {'YES' if len(b_gptq)+code_bytes < 16_000_000 else 'NO'}", flush=True)