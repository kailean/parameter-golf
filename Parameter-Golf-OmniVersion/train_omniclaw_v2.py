#!/usr/bin/env python3
"""
OMNICLAW-V2: "Singularity-S"
- Gumbel-Softmax Learnable Routing
- Sparse Factorized Embeddings (HYPERION)
- SWA (Stochastic Weight Averaging)
- Adaptive Depth Routing
- Target: < 0.60 BPB
"""

import os, sys, time, math, random, json, glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OmniConfig:
    vocab_size: int = 1024
    atom_dim: int = 64
    dict_size: int = 256
    sparsity_k: int = 4
    
    max_depth: int = 8
    depth_options: Tuple[int, int, int] = (2, 4, 8)
    
    num_heads: int = 8
    mlp_mult: int = 3
    
    iterations: int = 20000
    batch_tokens: int = 65536
    seq_len: int = 1024
    lr: float = 0.001
    swa_start_step: int = 18000
    
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    run_id: str = f"omniclaw_v2_{int(time.time())}"
    val_every: int = 500

# ═════════════════════════════════════════════════════════════════════════════
# COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════

class SparseEmbed(nn.Module):
    def __init__(self, vocab=1024, dict_size=256, atom_dim=64, k=4):
        super().__init__()
        self.dictionary = nn.Embedding(dict_size, atom_dim)
        self.indices = nn.Embedding(vocab, k)
        self.weights = nn.Embedding(vocab, k)
        self.dictionary.weight *= 0.02
        self.weights.weight *= 0.01
    
    def __call__(self, t):
        B, T = t.shape
        idx = self.indices(t).astype(mx.int32)
        w = nn.softmax(self.weights(t), axis=-1)
        atoms = self.dictionary(idx.reshape(-1)).reshape(B, T, -1, self.dictionary.weight.shape[1])
        return (mx.expand_dims(w, -1) * atoms).sum(axis=2)

class GumbelRouter(nn.Module):
    """Learnable Router using Gumbel-Softmax for differentiable routing"""
    def __init__(self, dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        self.depths = mx.array([2, 4, 8])
        self.temperature = 1.0

    def __call__(self, x, training=True):
        feat = x.mean(axis=1)
        logits = self.predictor(feat)
        
        if training:
            # Gumbel-Softmax trick for differentiable sampling
            gumbels = mx.log(-mx.log(mx.random.uniform(logits.shape)))
            soft_depths = nn.softmax((logits + gumbels) / self.temperature, axis=-1)
            # Expected depth for loss calculation
            depth = (soft_depths * self.depths).sum(axis=-1)
        else:
            depth = self.depths[mx.argmax(logits, axis=-1)]
            
        return depth

class OmniBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 3), nn.GELU(), nn.Linear(dim * 3, dim)
        )
    
    def __call__(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(self.ln1(x)).reshape(B, T, 3, 8, D//8)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        scores = (q @ k.transpose(0, 2, 1, 3)) / math.sqrt(D//8)
        if mask is not None: scores += mask
        attn = nn.softmax(scores, axis=-1) @ v
        x = x + self.out(attn.reshape(B, T, D))
        x = x + self.ffn(self.ln2(x))
        return x

class OmniClawV2(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config
        self.embed = SparseEmbed(config.vocab_size, config.dict_size, config.atom_dim, config.sparsity_k)
        self.router = GumbelRouter(config.atom_dim)
        self.blocks = [OmniBlock(config.atom_dim) for _ in range(config.max_depth)]
        self.ln_f = nn.LayerNorm(config.atom_dim)
        self.head = nn.Linear(config.atom_dim, config.vocab_size)
    
    def __call__(self, tokens, targets=None, training=True):
        x = self.embed(tokens)
        B, T, D = x.shape
        
        # ROUTING FIX: Ensure we get a native Python int for the loop
        depth_array = self.router(x, training=training)
        
        # Convert MLX array/scalar to native Python int safely
        if hasattr(depth_array, 'item'):
            exec_depth = int(depth_array.item())
        elif isinstance(depth_array, (np.ndarray, mx.array)):
            exec_depth = int(depth_array.flatten()[0])
        else:
            exec_depth = int(depth_array)
            
        exec_depth = max(2, min(exec_depth, self.config.max_depth))
        
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        for i in range(exec_depth):
            x = self.blocks[i](x, mask)
            
        logits = self.head(self.ln_f(x))
        
        if targets is not None:
            loss = nn.losses.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
            return loss, logits
        return logits

# ═════════════════════════════════════════════════════════════════════════════
# ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def train():
    config = OmniConfig()
    model = OmniClawV2(config)
    
    train_files = sorted(glob.glob(f"{config.data_path}/fineweb_train_*.bin"))
    val_file = f"{config.data_path}/fineweb_val_000000.bin"
    val_tokens = np.fromfile(val_file, dtype=np.uint32).astype(np.int32)
    
    opt = optim.Adam(learning_rate=config.lr)
    swa_model = None
    
    print(f"🚀 Starting OmniClaw-V2 | Target: < 0.60 BPB | Params: {sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters())):,}")

    for step in range(config.iterations):
        # Batch
        file = random.choice(train_files)
        tokens = np.fromfile(file, dtype=np.uint32).astype(np.int32)
        start = random.randint(0, max(1, len(tokens) - config.seq_len - 1))
        chunk = tokens[start:start + config.seq_len + 1]
        x, y = mx.array(chunk[:-1].reshape(1, -1)), mx.array(chunk[1:].reshape(1, -1))
        
        def loss_fn(m):
            l, _ = m(x, y, training=True)
            return l.mean()
            
        loss, grads = mx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        
        # SWA Logic
        if step >= config.swa_start_step:
            if swa_model is None:
                swa_model = OmniClawV2(config)
                # Initialize SWA with current weights
                swa_model.update(model.state)
            # Moving average of weights
            for p_swa, p_model in zip(swa_model.parameters(), model.parameters()):
                p_swa[:] = 0.99 * p_swa + 0.01 * p_model
                
        if step % 100 == 0:
            print(f"Step {step}/{config.iterations} | Loss: {float(loss):.4f}")

        if step > 0 and step % config.val_every == 0:
            model.eval()
            v_loss = 0.0
            for i in range(10):
                start = random.randint(0, len(val_tokens)-2049)
                vx = mx.array(val_tokens[start:start+2048].reshape(1, -1))
                vy = mx.array(val_tokens[start+1:start+2049].reshape(1, -1))
                l, _ = model(vx, vy, training=False)
                v_loss += float(l.mean())
            bpb = (v_loss/10) / math.log(2)
            print(f"--- Validation Step {step} | BPB: {bpb:.4f} ---")
            model.train()

    # Final Save
    mx.savez(f"{config.run_id}_final.npz", **{k:v for k,v in tree_flatten(model.state)})
    print(f"✅ Model saved to {config.run_id}_final.npz")

if __name__ == "__main__":
    train()
