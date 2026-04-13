#!/usr/bin/env python3
"""
MLX Evaluation Script for Parameter Golf
Loads a checkpoint and runs proper validation with sliding window eval.
"""
import os, sys, glob, math, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Import from main training script
from train_gpt_mlx_kl import (
    Hyperparameters, GPT, load_validation_tokens,
    load_sentencepiece_tokenizer, bigram_hash_logit_bias,
    sliding_window_cross_entropy_loss, tokenizer_aware_decode
)

def load_checkpoint(checkpoint_path: str, config: Hyperparameters):
    """Load model from MLX checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load weights
    weights = mx.load(checkpoint_path)
    
    # Create model
    model = GPT(config)
    
    # Convert PyTorch-style keys to MLX-style
    # PyTorch: blocks.0.attn.c_q.weight -> MLX: block_0.attn.c_q.weight
    mlx_weights = {}
    for key, value in weights.items():
        # Convert blocks.N. -> block_N.
        if key.startswith("blocks."):
            parts = key.split(".")
            layer_num = parts[1]
            new_key = f"block_{layer_num}.{'.'.join(parts[2:])}"
            mlx_weights[new_key] = value
        else:
            mlx_weights[key] = value
    
    # Update model
    model.update(tree_unflatten(list(mlx_weights.items())))
    
    print(f"Loaded {len(mlx_weights)} parameters")
    param_count = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Total parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    return model

def evaluate_validation(model, val_tokens, config, max_windows=None):
    """Run sliding window validation and compute BPB."""
    print(f"\nStarting validation...")
    print(f"  Sequence length: {config.eval_seq_len}")
    print(f"  Stride: {config.eval_stride}")
    print(f"  Batch size: {config.eval_batch_seqs} sequences")
    
    total_bytes = 0
    total_nll = 0.0
    window_count = 0
    start_time = time.time()
    
    seq_len = config.eval_seq_len
    stride = config.eval_stride
    batch_seqs = config.eval_batch_seqs
    
    # Calculate windows
    total_len = len(val_tokens)
    num_windows = (total_len - seq_len) // stride + 1
    
    if max_windows:
        num_windows = min(num_windows, max_windows)
    
    print(f"  Total windows: {num_windows:,}")
    
    for start_idx in range(0, num_windows * stride, stride * batch_seqs):
        batch_tokens = []
        for b in range(batch_seqs):
            s = start_idx + b * stride
            if s + seq_len > total_len:
                break
            batch_tokens.append(val_tokens[s:s + seq_len + 1])
        
        if len(batch_tokens) == 0:
            break
        
        # Stack batch
        batch = mx.stack([mx.array(t) for t in batch_tokens])
        B = batch.shape[0]
        
        # Forward pass
        logits, _ = model(batch[:, :-1], targets=batch[:, 1:])
        
        # Compute loss
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
            reduction='sum'
        )
        
        # Accumulate
        total_nll += float(loss.item())
        total_bytes += B * seq_len * 1  # Assuming 1 byte per token for now
        window_count += B
        
        # Progress
        if window_count % 100 == 0:
            elapsed = time.time() - start_time
            bpb = total_nll / (total_bytes * math.log(2))
            print(f"  Windows: {window_count:,}/{num_windows:,} | "
                  f"BPB: {bpb:.4f} | Time: {elapsed:.1f}s")
    
    # Final BPB
    total_bits = total_nll / math.log(2)
    final_bpb = total_bits / total_bytes if total_bytes > 0 else 0
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*50}")
    print(f"Windows processed: {window_count:,}")
    print(f"Total bytes: {total_bytes:,}")
    print(f"Total bits: {total_bits:,.0f}")
    print(f"Final val_bpb: {final_bpb:.4f}")
    print(f"Time: {elapsed:.1f}s ({window_count/elapsed:.1f} windows/s)")
    print(f"{'='*50}\n")
    
    return final_bpb

def main():
    config = Hyperparameters()
    config.eval_mode = "sliding"
    
    # Find checkpoint
    checkpoint_path = os.environ.get("CHECKPOINT", None)
    if not checkpoint_path:
        # Look in logs directory
        log_dir = Path("logs")
        checkpoints = list(log_dir.glob("*_model.npz"))
        if checkpoints:
            checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
            print(f"Found checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found. Set CHECKPOINT=path/to/model.npz")
            return
    
    # Load model
    model = load_checkpoint(checkpoint_path, config)
    
    # Load validation data
    print("\nLoading validation data...")
    val_tokens = load_validation_tokens(config.val_files, config.eval_seq_len)
    print(f"Loaded {len(val_tokens):,} validation tokens")
    
    # Run evaluation
    val_bpb = evaluate_validation(model, val_tokens, config)
    
    # Save result
    result_path = f"logs/val_bpb_{config.run_id}.txt"
    with open(result_path, "w") as f:
        f.write(f"val_bpb: {val_bpb:.6f}\n")
        f.write(f"checkpoint: {checkpoint_path}\n")
    print(f"Result saved to: {result_path}")

if __name__ == "__main__":
    main()