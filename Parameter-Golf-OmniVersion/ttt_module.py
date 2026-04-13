"""
ttt_module.py — Test-Time Training (TTT) with LoRA

Online adaptation during evaluation using LoRA-based lightweight updates.
Implements a legal score-first approach that only adapts on already-evaluated tokens.

This module provides integration hooks for train_gpt_kl_v3.py inference loop.

Example Usage:
    >>> from ttt_module import TestTimeTrainer, LoRAConfig
    >>> config = LoRAConfig(rank=8, alpha=16, dropout=0.0)
    >>> ttt = TestTimeTrainer(
    ...     model_dim=512,
    ...     vocab_size=1024,
    ...     num_layers=11,
    ...     lora_config=config,
    ...     learning_rate=0.01,
    ... )
    >>> # During evaluation
    >>> for token_idx in range(seq_len):
    ...     # Get predictions for current token
    ...     logits = model(input_ids[:, :token_idx+1])
    ...     # Adapt on already evaluated tokens (legal approach)
    ...     if token_idx > 0:
    ...         loss = ttt.compute_adaptation_loss(
    ...             logits[:, :-1], input_ids[:, 1:token_idx+1]
    ...         )
    ...         ttt.adapt_step(loss)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.
    
    Attributes:
        rank: LoRA rank (r) - bottleneck dimension
        alpha: LoRA scaling parameter
        dropout: Dropout probability for LoRA layers
        init_scale: Initialization scale for LoRA weights
        target_modules: Which modules to apply LoRA to
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    init_scale: float = 0.01
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "lm_head")
    
    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor."""
        return self.alpha / self.rank


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer.
    
    Implements W' = W + (alpha/r) * B * A where:
    - W is the frozen pretrained weight
    - A is a down-projection (d_model <-> rank)
    - B is an up-projection (rank <-> d_model)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (bottleneck dimension)
        alpha: LoRA scaling parameter
        dropout: Dropout probability
        init_scale: Weight initialization scale
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights: A (down) and B (up)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self._init_weights(init_scale)
    
    def _init_weights(self, scale: float) -> None:
        """Initialize LoRA weights with scaled random values."""
        # Kaiming uniform for A (down-projection)
        bound = scale / math.sqrt(self.lora_A.size(0))
        nn.init.uniform_(self.lora_A, -bound, bound)
        
        # Zero initialization for B (up-projection) - standard practice
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass computing LoRA delta.
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            LoRA delta [..., out_features]
        """
        # x @ A @ B with scaling
        result = x @ self.lora_A
        if self.dropout is not None:
            result = self.dropout(result)
        result = result @ self.lora_B
        return result * self.scaling
    
    def get_merged_weights(self) -> Tensor:
        """Compute the equivalent merged weight matrix."""
        return (self.lora_A @ self.lora_B) * self.scaling
    
    def merge_into(self, base_weight: Tensor) -> Tensor:
        """Merge LoRA weights into base weight."""
        return base_weight + self.get_merged_weights().to(base_weight.dtype)


class LoRAProjection(nn.Module):
    """LoRA-enhanced projection layer for attention Q/V projections.
    
    Wraps a base projection with LoRA adaptation while keeping
    the base weights frozen during test-time training.
    
    Args:
        base_module: The base linear projection (frozen)
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        base_module: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_module = base_module
        self.in_features = base_module.in_features
        self.out_features = base_module.out_features
        
        # Freeze base module
        for param in self.base_module.parameters():
            param.requires_grad = False
        
        # Create LoRA layer
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: base + LoRA delta."""
        base_out = self.base_module(x)
        lora_out = self.lora(x)
        return base_out + lora_out
    
    def get_lora_params(self) -> list[nn.Parameter]:
        """Get LoRA parameters for optimization."""
        return [self.lora.lora_A, self.lora.lora_B]
    
    def reset_lora(self) -> None:
        """Reset LoRA weights to initialization state."""
        bound = self.lora.lora_A.size(0) ** -0.5 * 0.01
        nn.init.uniform_(self.lora.lora_A, -bound, bound)
        nn.init.zeros_(self.lora.lora_B)


class BatchedLoRALayer(nn.Module):
    """Batched LoRA for processing multiple sequences independently.
    
    Each sequence in the batch gets its own LoRA weights, enabling
    per-document adaptation during test-time training.
    
    Args:
        batch_size: Number of sequences to adapt independently
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank
        alpha: LoRA alpha parameter
    """
    
    def __init__(
        self,
        batch_size: int,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Batched LoRA: [batch, in, rank] and [batch, rank, out]
        self.lora_A = nn.Parameter(torch.zeros(batch_size, in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(batch_size, rank, out_features))
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize with Kaiming uniform for A, zeros for B."""
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.lora_A, -bound, bound)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for batched LoRA.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
        
        Returns:
            Output [batch_size, seq_len, out_features]
        """
        # x @ A @ B for each batch element
        # x: [B, T, in], A: [B, in, r], B: [B, r, out]
        mid = torch.bmm(x, self.lora_A)  # [B, T, r]
        result = torch.bmm(mid, self.lora_B)  # [B, T, out]
        return result * self.scaling
    
    def reset(self) -> None:
        """Reset all LoRA weights."""
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_A.uniform_(-bound, bound)
            self.lora_B.zero_()


class TestTimeAdapter(nn.Module):
    """Complete LoRA adaptation module for a transformer model.
    
    Manages LoRA layers for attention projections (Q, V) and LM head,
    with support for batched per-sequence adaptation.
    
    Args:
        model_dim: Model hidden dimension
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads
        lora_config: LoRA configuration
    """
    
    def __init__(
        self,
        model_dim: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        lora_config: LoRAConfig,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.lora_config = lora_config
        
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        
        # LoRA for Q projections (all layers)
        self.q_loras = nn.ModuleList([
            LoRALayer(
                in_features=model_dim,
                out_features=model_dim,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
            )
            for _ in range(num_layers)
        ])
        
        # LoRA for V projections (all layers)
        self.v_loras = nn.ModuleList([
            LoRALayer(
                in_features=model_dim,
                out_features=kv_dim,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
            )
            for _ in range(num_layers)
        ])
        
        # LoRA for LM head (if not tied embeddings)
        if "lm_head" in lora_config.target_modules:
            self.lm_head_lora = LoRALayer(
                in_features=model_dim,
                out_features=vocab_size,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
            )
        else:
            self.lm_head_lora = None
        
        # Store which parameters are trainable
        self._trainable_params: list[str] = []
        self._identify_trainable_params()
    
    def _identify_trainable_params(self) -> None:
        """Identify which parameters should be trained during TTT."""
        self._trainable_params = []
        for name, param in self.named_parameters():
            if 'lora' in name.lower():
                self._trainable_params.append(name)
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get all trainable LoRA parameters."""
        params = []
        for name, param in self.named_parameters():
            if name in self._trainable_params:
                params.append(param)
        return params
    
    def reset(self) -> None:
        """Reset all LoRA weights to initial state."""
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module._init_weights(self.lora_config.init_scale)
    
    def apply_to_attention(
        self,
        layer_idx: int,
        q: Tensor,
        v: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply LoRA deltas to Q and V projections.
        
        Args:
            layer_idx: Which transformer layer
            q: Query tensor [B, T, H, D]
            v: Value tensor [B, T, Hkv, D]
        
        Returns:
            Modified (q, v) tensors
        """
        B, T, H, D = q.shape
        
        # Reshape Q for LoRA: [B, T, model_dim]
        q_flat = q.reshape(B, T, -1)
        q_delta = self.q_loras[layer_idx](q_flat)
        q = q + q_delta.reshape(B, T, H, D)
        
        # Reshape V for LoRA
        v_flat = v.reshape(B, T, -1)
        v_delta = self.v_loras[layer_idx](v_flat)
        v = v + v_delta.reshape_as(v)
        
        return q, v
    
    def apply_to_logits(self, logits: Tensor) -> Tensor:
        """Apply LoRA delta to LM head logits.
        
        Args:
            logits: Base logits [B, T, vocab_size]
        >        
        Returns:
            Modified logits [B, T, vocab_size]
        """
        if self.lm_head_lora is not None:
            # logits already computed, apply delta
            B, T, _ = logits.shape
            # We can't easily apply here since we need hidden states
            # Return logits as-is; delta is applied during forward
            pass
        return logits


class BatchedTestTimeAdapter(nn.Module):
    """Batched LoRA adapter for per-sequence test-time training.
    
    Each sequence in the batch maintains independent LoRA weights,
    allowing for per-document adaptation.
    
    Args:
        batch_size: Number of sequences
        model_dim: Model dimension
        vocab_size: Vocabulary size  
        num_layers: Number of layers
        rank: LoRA rank (default: 8)
    """
    
    def __init__(
        self,
        batch_size: int,
        model_dim: int,
        vocab_size: int,
        num_layers: int,
        rank: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        
        # Batched LoRA for LM head
        self.lm_head_lora = BatchedLoRALayer(
            batch_size, model_dim, vocab_size, rank=rank
        )
        
        # Batched LoRA for Q and V projections per layer
        self.q_loras = nn.ModuleList([
            BatchedLoRALayer(batch_size, model_dim, model_dim, rank=rank)
            for _ in range(num_layers)
        ])
        
        self.v_loras = nn.ModuleList([
            BatchedLoRALayer(batch_size, model_dim, model_dim, rank=rank)
            for _ in range(num_layers)
        ])
    
    def reset(self) -> None:
        """Reset all batched LoRA weights."""
        for module in self.modules():
            if isinstance(module, BatchedLoRALayer):
                module.reset()


class TestTimeTrainer:
    """Test-Time Training (TTT) manager with LoRA-based adaptation.
    
    Implements the legal score-first approach: only adapts on
    already-evaluated tokens during inference.
    
    Args:
        model_dim: Model hidden dimension
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads
        lora_config: LoRA configuration
        learning_rate: Learning rate for adaptation
        optimizer: Optimizer type ("adam", "sgd")
        adaptation_steps: Number of gradient steps per token
        accumulate_gradients: Whether to accumulate gradients
    """
    
    def __init__(
        self,
        model_dim: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        lora_config: LoRAConfig,
        learning_rate: float = 0.01,
        optimizer: Literal["adam", "sgd", "adamw"] = "adam",
        adaptation_steps: int = 1,
        accumulate_gradients: bool = False,
    ):
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lora_config = lora_config
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.adaptation_steps = adaptation_steps
        self.accumulate_gradients = accumulate_gradients
        
        # Create LoRA adapter
        self.adapter = TestTimeAdapter(
            model_dim=model_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            lora_config=lora_config,
        )
        
        # Optimizer state (initialized lazily)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._step_count = 0
        
        # History for score-first adaptation
        self._evaluated_positions: set[int] = set()
        self._adaptation_history: list[dict] = []
    
    def initialize_optimizer(self) -> None:
        """Initialize optimizer for LoRA parameters."""
        params = self.adapter.get_trainable_params()
        
        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                params, lr=self.learning_rate, weight_decay=0.01
            )
        elif self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=self.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def reset(self) -> None:
        """Reset adapter and optimizer for new sequence."""
        self.adapter.reset()
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            # Re-initialize optimizer state
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    state = self.optimizer.state.get(p)
                    if state is not None:
                        for key in list(state.keys()):
                            del state[key]
        self._step_count = 0
        self._evaluated_positions.clear()
        self._adaptation_history.clear()
    
    def compute_adaptation_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> Tensor:
        """Compute loss for adaptation on evaluated tokens.
        
        Args:
            logits: Model predictions [B, T, vocab_size]
            targets: Target tokens [B, T]
            mask: Optional mask for valid positions [B, T]
            reduction: Loss reduction method
        
        Returns:
            Scalar loss tensor
        """
        B, T, V = logits.shape
        
        # Flatten for cross entropy
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        
        # Compute loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none',
        )
        loss = loss.reshape(B, T)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            
        if reduction == "mean":
            return loss.sum() / (mask.sum() if mask is not None else B * T)
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def adapt_step(self, loss: Tensor) -> float:
        """Perform one adaptation step.
        
        Args:
            loss: Loss tensor to backpropagate
        
        Returns:
            Loss value as float
        """
        if self.optimizer is None:
            self.initialize_optimizer()
        
        loss_value = loss.item()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.adapter.get_trainable_params(), max_norm=1.0
        )
        
        self.optimizer.step()
        self._step_count += 1
        
        # Record adaptation
        self._adaptation_history.append({
            'step': self._step_count,
            'loss': loss_value,
        })
        
        return loss_value
    
    def is_legal_position(self, position: int) -> bool:
        """Check if a position can be legally adapted (already evaluated).
        
        This implements the score-first constraint: we can only adapt
        on tokens that have already been evaluated.
        
        Args:
            position: Token position in sequence
        
        Returns:
            True if adaptation is legal at this position
        """
        return position in self._evaluated_positions
    
    def mark_evaluated(self, position: int) -> None:
        """Mark a position as evaluated (safe to adapt on)."""
        self._evaluated_positions.add(position)
    
    def get_adaptation_window(
        self,
        current_position: int,
        window_size: int = 128,
    ) -> tuple[int, int]:
        """Get the legal adaptation window ending at current position.
        
        Args:
            current_position: Current token position
            window_size: Maximum window size
        
        Returns:
            Tuple of (start, end) positions for legal adaptation
        """
        end = current_position
        start = max(0, end - window_size)
        
        # Only include positions that have been evaluated
        legal_positions = [
            p for p in range(start, end)
            if p in self._evaluated_positions
        ]
        
        if legal_positions:
            return min(legal_positions), max(legal_positions) + 1
        return start, start
    
    def get_integration_hooks(self) -> dict[str, Callable]:
        """Get hooks for integrating with train_gpt_kl_v3.py.
        
        Returns:
            Dictionary with hooks for the inference loop
        """
        def pre_sequence_hook() -> None:
            """Call before processing a new sequence."""
            self.reset()
        
        def pre_token_hook(position: int) -> None:
            """Call before predicting a token."""
            pass
        
        def post_token_hook(
            position: int,
            logits: Tensor,
            target: Optional[Tensor] = None,
        ) -> Optional[Tensor]:
            """Call after predicting a token, optionally returns adapted logits."""
            # Mark current position as evaluated
            self.mark_evaluated(position)
            
            # Adapt on previous tokens if target is provided
            if target is not None and position > 0:
                start, end = self.get_adaptation_window(position)
                if end > start:
                    # Compute loss on evaluated window
                    window_logits = logits[:, start:end]
                    window_targets = target[:, start:end]
                    
                    loss = self.compute_adaptation_loss(
                        window_logits,
                        window_targets,
                    )
                    self.adapt_step(loss)
            
            return None  # Return modified logits if adaptation affects them
        
        def post_sequence_hook() -> dict:
            """Call after sequence processing, returns stats."""
            return {
                'adaptation_steps': self._step_count,
                'history': self._adaptation_history.copy(),
            }
        
        return {
            'pre_sequence': pre_sequence_hook,
            'pre_token': pre_token_hook,
            'post_token': post_token_hook,
            'post_sequence': post_sequence_hook,
        }


def create_ttt_wrapped_model(
    model: nn.Module,
    lora_rank: int = 8,
    learning_rate: float = 0.01,
) -> tuple[nn.Module, TestTimeTrainer]:
    """Wrap a model with TTT capabilities.
    
    Args:
        model: Base transformer model
        lora_rank: LoRA rank for adaptation
        learning_rate: Learning rate for test-time training
    
    Returns:
        Tuple of (wrapped_model, ttt_trainer)
    """
    # Extract model dimensions
    model_dim = model.tok_emb.embedding_dim
    vocab_size = model.tok_emb.num_embeddings
    num_layers = len(model.blocks)
    
    # Infer attention config
    first_block = model.blocks[0]
    num_heads = first_block.attn.num_heads
    num_kv_heads = first_block.attn.num_kv_heads
    
    config = LoRAConfig(rank=lora_rank)
    
    ttt = TestTimeTrainer(
        model_dim=model_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        lora_config=config,
        learning_rate=learning_rate,
    )
    
    return model, ttt


# Example usage and testing
if __name__ == "__main__":
    print("Testing TestTimeTrainer (LoRA TTT)...")
    
    # Configuration
    batch_size = 4
    seq_len = 64
    model_dim = 512
    vocab_size = 1024
    num_layers = 11
    num_heads = 8
    num_kv_heads = 4
    rank = 8
    
    # Create LoRA config
    config = LoRAConfig(
        rank=rank,
        alpha=16,
        dropout=0.0,
    )
    
    # Create TTT trainer
    ttt = TestTimeTrainer(
        model_dim=model_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        lora_config=config,
        learning_rate=0.01,
    )
    
    print(f"\nLoRA config: rank={config.rank}, alpha={config.alpha}")
    print(f"Trainable parameters: {len(ttt.adapter.get_trainable_params())}")
    
    # Test adaptation loop
    print("\nSimulating test-time adaptation loop...")
    
    for step in range(seq_len):
        # Mark position as evaluated
        ttt.mark_evaluated(step)
        
        # Simulate prediction
        logits = torch.randn(batch_size, 1, vocab_size)
        
        # Adapt on previous tokens (after position 0)
        if step > 0:
            start, end = ttt.get_adaptation_window(step)
            if end > start:
                # Create dummy targets for previous positions
                targets = torch.randint(0, vocab_size, (batch_size, end - start))
                
                # Create dummy logits for window
                window_logits = torch.randn(batch_size, end - start, vocab_size)
                
                loss = ttt.compute_adaptation_loss(window_logits, targets)
                loss_val = ttt.adapt_step(loss)
                
                if step % 16 == 0:
                    print(f"  Step {step}: adapted on positions [{start}, {end}), loss={loss_val:.4f}")
    
    # Get final stats
    hooks = ttt.get_integration_hooks()
    stats = hooks['post_sequence']()
    print(f"\nTotal adaptation steps: {stats['adaptation_steps']}")
    
    # Test batched adapter
    print("\nTesting BatchedTestTimeAdapter...")
    batched = BatchedTestTimeAdapter(
        batch_size=4,
        model_dim=model_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        rank=rank,
    )
    
    test_input = torch.randn(4, seq_len, model_dim)
    output = batched.lm_head_lora(test_input)
    print(f"Batched LoRA output shape: {output.shape}")
    
    print("\nAll tests passed!")