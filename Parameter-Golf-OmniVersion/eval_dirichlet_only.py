#!/usr/bin/env python3
"""Eval-only: load saved int6-zstd checkpoint and run Dirichlet sliding eval."""
import sys, os, math, time, pickle
import numpy as np
import zstandard
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from pathlib import Path
import sentencepiece as spm
from collections import defaultdict

# ── Import the model + quantization code from the training script ──
sys.path.insert(0, os.path.dirname(__file__))
# We'll exec the relevant classes from train_gpt_mlx_kl.py

# Load everything from the training script (classes only, no main())
exec_globals = {}
with open(os.path.join(os.path.dirname(__file__), "train_gpt_mlx_kl.py")) as f:
    source = f.read()

# Extract just the class/function definitions (everything before main())
# We'll import the whole thing but skip main() by not calling it
exec(compile(source, "train_gpt_mlx_kl.py", "exec"), exec_globals)

# Provide a module-level log function (referenced inside eval_val_sliding_ngram)
def _log(msg, console=True):
    if console:
        print(msg, flush=True)
exec_globals["log"] = _log

# Now we have all classes available
GPT = exec_globals["GPT"]
Hyperparameters = exec_globals["Hyperparameters"]
dequantize_state_dict_int6 = exec_globals["dequantize_state_dict_int6"]
DirichletNgramMixer = exec_globals["DirichletNgramMixer"]
build_sentencepiece_luts = exec_globals["build_sentencepiece_luts"]
load_validation_tokens = exec_globals["load_validation_tokens"]

def main():
    args = Hyperparameters()
    # Override for Dirichlet eval
    args.ngram_mixer_enabled = True
    args.dirichlet_mixer = True
    args.dirichlet_max_order = 15
    args.dirichlet_alpha = 0.25
    args.phrase_cache_enabled = True
    args.phrase_probe_lengths = "20,16"
    args.eval_seq_len = 2048
    args.eval_stride = 64
    args.eval_batch_seqs = 32

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    print(f"Loading validation data from {args.val_files}...")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    # Build model
    print("Building model...")
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init, bigram_hash_size=args.bigram_hash_size,
        use_ortho_init=args.use_ortho_init,
        rope_dims=args.rope_dims, xsa_last_n=args.xsa_last_n,
        use_ln_scale=args.ln_scale_enabled, smear_enabled=args.smear_enabled,
        engram_lite_enabled=args.engram_lite_enabled,
        engram_hash_size=args.engram_hash_size,
        engram_embed_dim=args.engram_embed_dim,
        engram_n_heads=args.engram_n_heads,
        skipgram_hash_size=args.skipgram_hash_size,
    )
    model.use_qat = False

    # Load int6-zstd checkpoint
    checkpoint_path = Path("logs/dirichlet_smoke/dirichlet_smoke_mlx_model.int6.ptz")
    print(f"Loading quantized checkpoint from {checkpoint_path}...")
    t0 = time.perf_counter()
    with checkpoint_path.open("rb") as f:
        quant_blob = f.read()
    quant_obj = pickle.loads(zstandard.ZstdDecompressor().decompress(quant_blob))
    flat_state = dequantize_state_dict_int6(quant_obj)
    model.update(tree_unflatten(list(flat_state.items())))
    mx.eval(model.parameters())
    print(f"Checkpoint loaded in {1000*(time.perf_counter()-t0):.0f}ms")

    # Run Dirichlet sliding eval
    print(f"\n{'='*60}")
    print(f"DirichletNgramMixer Sliding Eval")
    print(f"  max_order={args.dirichlet_max_order}")
    print(f"  alpha={args.dirichlet_alpha}")
    print(f"  phrase_cache={args.phrase_cache_enabled}")
    print(f"  probe_lengths={args.phrase_probe_lengths}")
    print(f"  eval_seq_len={args.eval_seq_len}")
    print(f"  stride={args.eval_stride}")
    print(f"  batch_seqs={args.eval_batch_seqs}")
    print(f"{'='*60}\n")

    # Monkey-patch the eval function to handle bfloat16 logits
    # The original has: logits_np = np.array(logits_all) which fails on bfloat16
    import types
    
    original_eval_fn = exec_globals["eval_val_sliding_ngram"]
    
    # We need to override the model's token_logits to return float32
    original_token_logits = model.token_logits
    
    def token_logits_float32(input_ids):
        result = original_token_logits(input_ids)
        return result.astype(mx.float32)
    
    model.token_logits = token_logits_float32

    eval_fn = original_eval_fn

    def log(msg):
        print(msg, flush=True)

    t0 = time.perf_counter()
    val_loss, val_bpb = eval_fn(args, model, val_tokens,
                                 base_bytes_lut, has_leading_space_lut,
                                 is_boundary_token_lut, log_fn=log)
    elapsed_ms = 1000 * (time.perf_counter() - t0)

    print(f"\n{'='*60}")
    print(f"FINAL: dirichlet_sliding val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} eval_time={elapsed_ms:.0f}ms")
    print(f"FINAL_EXACT: dirichlet_sliding val_loss={val_loss:.8f} val_bpb={val_bpb:.8f}")
    print(f"{'='*60}")

    # Also log baseline for comparison
    print(f"\nBaseline (neural-only) sliding bpb was ~3.90")
    print(f"Dirichlet improvement: {3.90 - val_bpb:.4f} BPB")

if __name__ == "__main__":
    main()