#!/usr/bin/env python3
"""
HYPERION: Hierarchical Yielding Programmable Encoding with Recurrent Inference ON-the-fly
Simplified version for Parameter Golf - focusing on sparse embeddings + hierarchical states.
"""

from __future__ import annotations
import os, sys, time, math
from dataclasses import dataclass
from typing import Optional
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class HyperionConfig:
    """HYPERION configuration"""
    vocab_size: int = 1024
    atom_dim: int = 64
    dict_size: int = 256
    sparsity_k: int = 4
    num_layers: int = 8  # Reduced for size
    mlp_mult: int = 3
    seq_len: int = 1024
    
    # Hierarchical
    token_dim: int = 64
    phrase_dim: int = 64
    clause_dim: int = 64
    doc_dim: int = 64
    
    # Training
    lr: float = 0.001


class SparseFactorizedEmbedding(nn.Module):
    """Sparse factorized embedding - 8x smaller than dense"""
    def __init__(self, vocab_size: int, dict_size: int, atom_dim: int, k: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dict_size = dict_size
        self.atom_dim = atom_dim
        self.k = k
        
        # Dictionary of atoms
        self.dictionary = nn.Embedding(dict_size, atom_dim)
        # Token -> atom indices
        self.token_indices = nn.Embedding(vocab_size, k)
        # Token -> combination weights
        self.token_weights = nn.Embedding(vocab_size, k)
        
        # Init
        self.dictionary.weight = self.dictionary.weight * 0.02
        self.token_weights.weight = self.token_weights.weight * 0.01
    
    def __call__(self, tokens: mx.array) -> mx.array:
        B, T = tokens.shape
        
        # Get indices and weights
        indices = self.token_indices(tokens).astype(mx.int32)  # (B, T, k)
        weights = nn.softmax(self.token_weights(tokens), axis=-1)  # (B, T, k)
        
        # Gather atoms (flatten for gather)
        indices_flat = indices.reshape(-1)
        atoms = self.dictionary(indices_flat)  # (B*T*k, atom_dim)
        atoms = atoms.reshape(B, T, self.k, self.atom_dim)
        
        # Weighted sum
        weights = mx.expand_dims(weights, -1)  # (B, T, k, 1)
        embeddings = mx.sum(weights * atoms, axis=2)  # (B, T, atom_dim)
        
        return embeddings
    
    def count_params(self) -> int:
        return (self.dict_size * self.atom_dim + 
                self.vocab_size * self.k * 2)


class HierarchicalAttention(nn.Module):
    """Simplified hierarchical attention with 4-level states"""
    def __init__(self, config: HyperionConfig):
        super().__init__()
        self.config = config
        d = config.atom_dim
        
        # Projections for each level
        self.token_qkv = nn.Linear(d, d * 3)
        self.phrase_proj = nn.Linear(d, d)
        self.clause_proj = nn.Linear(d, d)
        self.doc_proj = nn.Linear(d, d)
        
        # Output
        self.out = nn.Linear(d * 4, d)
        
        # Gates
        self.gates = nn.Linear(d, 4)
    
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, d = x.shape
        
        # Token-level (standard attention)
        qkv = self.token_qkv(x)  # (B, T, d*3)
        q, k, v = qkv.split(3, axis=-1)
        
        # Simple attention
        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(d)
        if mask is not None:
            scores = scores + mask
        attn = nn.softmax(scores, axis=-1)
        token_out = attn @ v  # (B, T, d)
        
        # Higher levels (simplified - just projections with gating)
        phrase_out = self.phrase_proj(x)
        clause_out = self.clause_proj(x)
        doc_out = self.doc_proj(x)
        
        # Gating
        gate_logits = self.gates(x.mean(axis=1, keepdims=True))  # (B, 1, 4)
        gates = nn.softmax(gate_logits, axis=-1)  # (B, 1, 4)
        
        # Combine with gates
        combined = mx.concatenate([
            token_out,
            phrase_out,
            clause_out,
            doc_out
        ], axis=-1)  # (B, T, d*4)
        
        # Apply gates
        gates = mx.expand_dims(gates, 2)  # (B, 1, 1, 4)
        combined = combined.reshape(B, T, 4, d)
        out = mx.sum(combined * gates, axis=2)  # (B, T, d)
        
        return self.out(mx.concatenate([token_out, phrase_out, clause_out, doc_out], axis=-1))


class HyperionBlock(nn.Module):
    """HYPERION transformer block"""
    def __init__(self, config: HyperionConfig):
        super().__init__()
        d = config.atom_dim
        
        self.ln1 = nn.LayerNorm(d)
        self.attn = HierarchicalAttention(config)
        self.ln2 = nn.LayerNorm(d)
        
        # Small FFN
        self.ffn = nn.Sequential(
            nn.Linear(d, d * config.mlp_mult),
            nn.GELU(),
            nn.Linear(d * config.mlp_mult, d)
        )
    
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Attention with residual
        x = x + self.attn(self.ln1(x), mask)
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class HyperionGPT(nn.Module):
    """Complete HYPERION model"""
    def __init__(self, config: HyperionConfig):
        super().__init__()
        self.config = config
        
        # Sparse embedding
        self.embed = SparseFactorizedEmbedding(
            config.vocab_size, config.dict_size, 
            config.atom_dim, config.sparsity_k
        )
        
        # Blocks
        self.blocks = [HyperionBlock(config) for _ in range(config.num_layers)]
        
        # Output
        self.ln_final = nn.LayerNorm(config.atom_dim)
        self.lm_head = nn.Linear(config.atom_dim, config.vocab_size)
    
    def __call__(self, tokens: mx.array, targets: Optional[mx.array] = None):
        # Embed
        x = self.embed(tokens)
        B, T, _ = x.shape
        
        # Causal mask
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1)
            )
            return loss, logits
        
        return logits
    
    def count_params(self) -> int:
        return sum(int(np.prod(p.shape)) for _, p in tree_flatten(self.parameters()))


def main():
    print("=" * 60)
    print("HYPERION - Sparse Factorized GPT")
    print("=" * 60)
    
    config = HyperionConfig()
    
    print("\nConfiguration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Atom dim: {config.atom_dim}")
    print(f"  Dictionary size: {config.dict_size}")
    print(f"  Sparsity k: {config.sparsity_k}")
    print(f"  Layers: {config.num_layers}")
    
    # Create model
    print("\nBuilding model...")
    model = HyperionGPT(config)
    
    n_params = model.count_params()
    print(f"\nTotal parameters: {n_params:,}")
    print(f"Estimated size: {n_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"Target: <16 MB")
    
    # Compare embeddings
    standard = config.vocab_size * 512
    sparse = config.vocab_size * config.sparsity_k * 2 + config.dict_size * config.atom_dim
    print(f"\nEmbedding comparison:")
    print(f"  Standard: {standard:,} params")
    print(f"  Sparse:   {sparse:,} params")
    print(f"  Savings:  {standard/sparse:.1f}x")
    
    # Test
    print("\nTesting forward pass...")
    test_tokens = mx.random.randint(0, config.vocab_size, (2, 32))
    logits = model(test_tokens)
    print(f"✓ Forward pass OK: {test_tokens.shape} -> {logits.shape}")
    
    print("\n" + "=" * 60)
    print("HYPERION ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
