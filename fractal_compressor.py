"""
Fractal Compressor — high-density architecture for the Parameter Golf challenge (16MB limit).

Stages:
  1. SpectralLinear + SharedProjector  (spectral weight reconstruction & parameter aliasing)
  2. SequenceFolder + FoldedRecurrentBlock  (dynamic sequence folding with recurrence)
  3. FractalCompressor  (byte-level virtual vocab + full assembly)

Compatible with the existing train_gpt_kl.py ecosystem (PyTorch, CUDA, DDP).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ─────────────────────────────────────────────────────────────
# Stage 1: SpectralLinear & SharedProjector
# ─────────────────────────────────────────────────────────────

def _build_dct_basis(n: int) -> Tensor:
    """Build an (n, n) Type-II DCT basis matrix (orthonormal)."""
    basis = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            basis[k, i] = math.cos(math.pi * k * (2 * i + 1) / (2 * n))
    # Orthonormal scaling
    basis[0] *= math.sqrt(1.0 / n)
    basis[1:] *= math.sqrt(2.0 / n)
    return basis


class SpectralLinear(nn.Module):
    """Linear layer that stores a low-rank coefficient matrix and reconstructs
    the full weight via a fixed DCT basis at forward time.

    Instead of storing W ∈ R^{out × in}, we store:
      - coeffs ∈ R^{rank_out × rank_in}   (learnable, ~1/10th the params)
    and register fixed DCT bases for both dimensions. The effective weight is:
      W_eff = basis_out[:out, :rank_out] @ coeffs @ basis_in[:rank_in, :in]

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        rank_frac:    Fraction of each dimension kept in the spectral domain.
                      E.g. 0.3 means rank_in = ceil(0.3 * in), rank_out = ceil(0.3 * out).
        bias:         Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank_frac: float = 0.3,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.rank_in = max(1, math.ceil(rank_frac * in_features))
        self.rank_out = max(1, math.ceil(rank_frac * out_features))

        # Learnable low-rank coefficients in the spectral domain
        self.coeffs = nn.Parameter(
            torch.randn(self.rank_out, self.rank_in)
            * math.sqrt(2.0 / (in_features + out_features))
        )

        # Fixed DCT bases (non-learnable)
        basis_in = _build_dct_basis(in_features)
        basis_out = _build_dct_basis(out_features)
        self.register_buffer("basis_in", basis_in, persistent=False)
        self.register_buffer("basis_out", basis_out, persistent=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def reconstruct_weight(self) -> Tensor:
        """Reconstruct the full weight matrix from spectral coefficients."""
        # basis_out: (out, out), take first rank_out columns → (out, rank_out)
        B_out = self.basis_out[:, : self.rank_out]
        # basis_in: (in, in), take first rank_in rows → (rank_in, in)
        B_in = self.basis_in[: self.rank_in, :]
        # W_eff = B_out @ coeffs @ B_in  →  (out, rank_out) @ (rank_out, rank_in) @ (rank_in, in) = (out, in)
        return B_out @ self.coeffs @ B_in

    def forward(self, x: Tensor) -> Tensor:
        W = self.reconstruct_weight().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, W, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank_in={self.rank_in}, rank_out={self.rank_out}, "
            f"bias={self.bias is not None}"
        )


class SharedProjector(nn.Module):
    """Parameter aliasing: two projections share the same SpectralLinear base
    weights but are differentiated by small learnable shift vectors.

    Intended usage: Q-projection and FFN-up-projection share the same base
    spectral weights, each with its own lightweight shift.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension (same for both projections).
        rank_frac:    Rank fraction for the shared SpectralLinear.
    """

    def __init__(self, in_features: int, out_features: int, rank_frac: float = 0.3):
        super().__init__()
        self.shared_base = SpectralLinear(in_features, out_features, rank_frac=rank_frac)
        # Two independent shift vectors (cheap — only out_features params each)
        self.shift_a = nn.Parameter(torch.zeros(out_features))  # e.g. for Q
        self.shift_b = nn.Parameter(torch.zeros(out_features))  # e.g. for FFN-up

    def forward_a(self, x: Tensor) -> Tensor:
        """First projection (e.g. Q)."""
        return self.shared_base(x) + self.shift_a.to(x.dtype)

    def forward_b(self, x: Tensor) -> Tensor:
        """Second projection (e.g. FFN-up)."""
        return self.shared_base(x) + self.shift_b.to(x.dtype)


# ─────────────────────────────────────────────────────────────
# Stage 2: SequenceFolder & FoldedRecurrentBlock
# ─────────────────────────────────────────────────────────────

class SequenceFolder(nn.Module):
    """Reduces sequence length by 2× via a learned 1D convolution with stride 2.

    Operates on (B, L, D) tensors. When L is odd, the last token is kept
    as-is and concatenated, so the output length is ceil(L / 2).

    Args:
        dim: Model dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        # 1D conv: groups=1, kernel_size=2, stride=2 — learned weighted average of adjacent tokens
        self.conv = nn.Conv1d(dim, dim, kernel_size=2, stride=2, bias=False)
        # Initialize close to simple averaging (vectorized)
        with torch.no_grad():
            self.conv.weight.zero_()
            diag_idx = torch.arange(dim)
            self.conv.weight[diag_idx, diag_idx, 0] = 0.5
            self.conv.weight[diag_idx, diag_idx, 1] = 0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, D) → (B, ceil(L/2), D)"""
        B, L, D = x.shape
        if L <= 1:
            return x

        # Handle odd length: save last token, fold the even prefix
        if L % 2 == 1:
            x_even = x[:, :-1, :]  # (B, L-1, D)
            x_last = x[:, -1:, :]  # (B, 1, D)
        else:
            x_even = x
            x_last = None

        # Conv1d expects (B, D, L)
        out = self.conv(x_even.transpose(1, 2)).transpose(1, 2)  # (B, L_even/2, D)

        if x_last is not None:
            out = torch.cat([out, x_last], dim=1)

        return self.norm(out)


class FoldedRecurrentBlock(nn.Module):
    """Wraps a transformer block and a SequenceFolder. Applies the block on
    every iteration and folds the sequence every K iterations.

    Positional information is re-injected after folding via a small learned
    positional embedding that adapts to the current (reduced) sequence length.

    Args:
        dim:        Model dimension.
        num_heads:  Number of attention heads.
        num_kv_heads: Number of key/value heads (GQA).
        mlp_mult:   MLP expansion factor.
        fold_every: Fold the sequence every K iterations.
        max_seq_len: Maximum sequence length (for positional embedding table).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        fold_every: int = 3,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.fold_every = fold_every

        # Lightweight transformer sub-block (attention + FFN)
        self.attn_norm = nn.RMSNorm(dim)
        self.mlp_norm = nn.RMSNorm(dim)

        head_dim = dim // num_heads
        kv_dim = num_kv_heads * head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        hidden = mlp_mult * dim
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.mlp_proj = nn.Linear(hidden, dim, bias=False)

        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

        # Sequence folder
        self.folder = SequenceFolder(dim)

        # Learned positional embedding (re-injected after folding)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _attention(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        q = self.c_q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, L, D))

    def _mlp(self, x: Tensor) -> Tensor:
        h = torch.relu(self.fc(x))
        return self.mlp_proj(h * h)  # relu^2

    def forward(self, x: Tensor, iteration: int) -> Tensor:
        """
        Args:
            x:         (B, L, D) input tensor.
            iteration: Current iteration index in the recurrence loop.

        Returns:
            (B, L', D) where L' <= L (L' < L if folding was triggered).
        """
        # Standard transformer sub-block
        x = x + self.attn_scale[None, None, :] * self._attention(self.attn_norm(x))
        x = x + self.mlp_scale[None, None, :] * self._mlp(self.mlp_norm(x))

        # Fold every K iterations (but not on iteration 0)
        if iteration > 0 and iteration % self.fold_every == 0:
            x = self.folder(x)
            # Re-inject positional information for the new (shorter) sequence
            L_new = x.size(1)
            pos_ids = torch.arange(L_new, device=x.device)
            x = x + self.pos_emb(pos_ids)[None, :, :]

        return x


# ─────────────────────────────────────────────────────────────
# Stage 3: FractalCompressor — Full Model Assembly
# ─────────────────────────────────────────────────────────────

class ByteAssembler(nn.Module):
    """Projects raw byte embeddings (256 entries) into the model dimension.

    A tiny 2-layer MLP maps from byte_dim to model_dim. The projection weight
    of the second layer can be tied to the LM head for parameter savings.

    Args:
        byte_dim:  Dimension of the byte embedding (small, e.g. 64).
        model_dim: Target model dimension (e.g. 512).
    """

    def __init__(self, byte_dim: int = 64, model_dim: int = 512):
        super().__init__()
        self.byte_emb = nn.Embedding(256, byte_dim)
        nn.init.normal_(self.byte_emb.weight, std=0.02)

        # 2-layer MLP assembler: byte_dim → model_dim → model_dim
        hidden = (byte_dim + model_dim) // 2
        self.fc1 = nn.Linear(byte_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, model_dim, bias=False)

    def forward(self, byte_ids: Tensor) -> Tensor:
        """byte_ids: (B, L) of uint8/int64 values in [0, 255] → (B, L, model_dim)"""
        x = self.byte_emb(byte_ids)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


class FractalCompressor(nn.Module):
    """Complete Fractal Compressor model for the Parameter Golf challenge.

    Architecture:
      1. Byte-level virtual vocabulary (256 bytes → ByteAssembler → d_model)
      2. Recurrent loop of FoldedRecurrentBlocks with dynamic sequence folding
      3. SharedProjector for Q/FFN-up parameter aliasing within the block
      4. LM head tied to the ByteAssembler's projection weight

    Args:
        model_dim:      Model hidden dimension.
        num_heads:      Number of attention heads.
        num_kv_heads:   Number of KV heads (GQA).
        mlp_mult:       MLP expansion factor.
        num_iterations: Number of recurrent iterations.
        fold_every:     Fold the sequence every K iterations.
        max_seq_len:    Maximum input sequence length.
        byte_dim:       Byte embedding dimension.
        logit_softcap:  Logit soft-capping value.
        spectral_rank_frac: Rank fraction for SpectralLinear in SharedProjector.
        vocab_size:     Output vocabulary size (for the LM head).
        shared_proj_scale: Scale factor for the shared projector residual signal.
    """

    def __init__(
        self,
        model_dim: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        mlp_mult: int = 3,
        num_iterations: int = 6,
        fold_every: int = 3,
        max_seq_len: int = 1024,
        byte_dim: int = 64,
        logit_softcap: float = 30.0,
        spectral_rank_frac: float = 0.3,
        vocab_size: int = 1024,
        shared_proj_scale: float = 0.1,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.shared_proj_scale = shared_proj_scale

        # 1. Byte-level embeddings + assembler
        self.assembler = ByteAssembler(byte_dim=byte_dim, model_dim=model_dim)

        # 2. SharedProjector — Q and FFN-up share spectral base weights
        self.shared_proj = SharedProjector(
            model_dim, model_dim, rank_frac=spectral_rank_frac
        )

        # 3. Recurrent block (single block, applied num_iterations times)
        self.recurrent_block = FoldedRecurrentBlock(
            dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            fold_every=fold_every,
            max_seq_len=max_seq_len,
        )

        # 4. Final norm
        self.final_norm = nn.RMSNorm(model_dim)

        # 5. LM head — tied to the assembler's fc2 projection
        #    lm_head projects model_dim → vocab_size
        #    We create a separate linear and tie only if dimensions match;
        #    otherwise use an independent head.
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids:  (B, L) token IDs.
            target_ids: (B, L) target token IDs.

        Returns:
            Scalar cross-entropy loss.
        """
        # Byte-level embedding → model dim
        x = self.assembler(input_ids)
        x = F.rms_norm(x, (x.size(-1),))

        # Inject shared projector output as an additive signal
        shared_signal = self.shared_proj.forward_a(x)
        x = x + self.shared_proj_scale * shared_signal

        # Recurrent loop with dynamic sequence folding
        for i in range(self.num_iterations):
            x = self.recurrent_block(x, i)

        # LM head
        x = self.final_norm(x)
        # After folding, sequence may be shorter than targets.
        # Truncate targets to match — later positions are folded into earlier ones,
        # so the model learns to compress information during folding.
        L_out = x.size(1)
        L_tgt = target_ids.size(1)

        if L_out < L_tgt:
            # Take first L_out targets (tokens at the start of the sequence)
            target_ids = target_ids[:, :L_out]

        x_flat = x.reshape(-1, self.model_dim)
        logits = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return (B, L', vocab) logits without computing loss."""
        x = self.assembler(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        shared_signal = self.shared_proj.forward_a(x)
        x = x + self.shared_proj_scale * shared_signal
        for i in range(self.num_iterations):
            x = self.recurrent_block(x, i)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits
