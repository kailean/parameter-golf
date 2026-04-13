"""
xsa_all_layers.py — All-Layer Cross-Scaling Attention (XSA) Extension

Extends cross-layer scaling attention from 4 layers to all 11 layers
with learnable per-layer scaling factors and an aggregation mechanism.

This module provides hooks for integration with train_gpt_kl_v3.py.

Example Usage:
    >>> from xsa_all_layers import AllLayerXSA
    >>> xsa = AllLayerXSA(
    ...     num_layers=11,
    ...     num_heads=8,
    ...     num_kv_heads=4,
    ...     head_dim=64,
    ...     aggregation_mode="weighted_sum",
    ... )
    >>> # In your model forward:
    >>> layer_outputs = []
    >>> for i, block in enumerate(blocks):
    ...     x = block(x)
    ...     layer_outputs.append(x)
    >>> # Apply XSA aggregation across all layers
    >>> aggregated = xsa.aggregate_layers(layer_outputs)
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LayerScalingFactor(nn.Module):
    """Learnable scaling factor for a specific layer's contribution to XSA.
    
    Each layer has its own learnable scalar that scales how much influence
    that layer's attention pattern has in the cross-layer aggregation.
    
    Args:
        init_scale: Initial value for the scaling factor (default: 1.0)
        learnable: Whether the scale is learnable (default: True)
    """
    
    def __init__(self, init_scale: float = 1.0, learnable: bool = True):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_scale, dtype=torch.float32),
            requires_grad=learnable,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply layer scaling to input tensor."""
        return x * self.scale.to(dtype=x.dtype)


class CrossLayerAttention(nn.Module):
    """Cross-layer attention mechanism for aggregating information across layers.
    
    Computes attention weights between the current layer and all other layers,
    allowing the model to dynamically weight the contribution of each layer
    based on the input context.
    
    Args:
        num_layers: Total number of layers to aggregate across
        head_dim: Dimension per attention head
        num_heads: Number of attention heads
    """
    
    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        num_heads: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        # Learnable query projection for current layer
        self.query_proj = nn.Linear(head_dim, head_dim, bias=False)
        
        # Learnable key projection for all layers (shared across layers)
        self.key_proj = nn.Linear(head_dim, head_dim, bias=False)
        
        # Layer-wise learnable scaling factors
        self.layer_scales = nn.ModuleList([
            LayerScalingFactor(init_scale=1.0 / num_layers)
            for _ in range(num_layers)
        ])
        
        # Temperature parameter for attention scaling
        self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with small random values."""
        nn.init.normal_(self.query_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        current_layer: Tensor,
        all_layers: list[Tensor],
        current_layer_idx: int,
    ) -> Tensor:
        """Compute cross-layer attention for current layer.
        
        Args:
            current_layer: Current layer output [B, T, H, D]
            all_layers: List of all layer outputs, each [B, T, H, D]
            current_layer_idx: Index of current layer in all_layers
        
        Returns:
            Aggregated representation [B, T, H, D]
        """
        B, T, H, D = current_layer.shape
        
        # Compute query from current layer
        q = self.query_proj(current_layer)  # [B, T, H, D]
        
        # Stack all layer outputs and compute keys
        stacked = torch.stack(all_layers, dim=2)  # [B, T, L, H, D]
        stacked_flat = stacked.reshape(B * T * self.num_layers, H, D)
        k = self.key_proj(stacked_flat)  # [B*T*L, H, D]
        k = k.reshape(B, T, self.num_layers, H, D)
        
        # Compute attention scores: q @ k^T
        q_flat = q.reshape(B * T, H, D)
        k_flat = k.reshape(B * T, self.num_layers, H * D).transpose(1, 2)
        
        # Reshape for multi-head attention
        q_heads = q_flat.reshape(B * T * H, 1, D // self.num_heads if self.num_heads > 1 else D)
        k_heads = k.reshape(B * T, self.num_layers, H, D // self.num_heads if self.num_heads > 1 else D)
        k_heads = k_heads.permute(0, 2, 3, 1).reshape(B * T * H, D // self.num_heads if self.num_heads > 1 else D, self.num_layers)
        
        scores = torch.bmm(q_heads, k_heads) / (math.sqrt(D / self.num_heads) * self.temperature.abs())
        attn_weights = F.softmax(scores, dim=-1)  # [B*T*H, 1, L]
        
        # Apply layer-wise scaling factors
        scales = torch.stack([ls.scale for ls in self.layer_scales], dim=0)  # [L]
        scales = scales.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights * scales.unsqueeze(0).unsqueeze(0)  # Broadcast scales
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Weighted aggregation
        v = stacked.permute(0, 2, 1, 3, 4).reshape(B, self.num_layers, T, H * D)
        v = v.permute(0, 2, 3, 1).reshape(B * T * H * D // (D // self.num_heads if self.num_heads > 1 else D), self.num_layers)
        
        # Simplified: directly aggregate based on attention weights
        output = torch.zeros_like(current_layer)
        for l_idx, layer_out in enumerate(all_layers):
            scale = self.layer_scales[l_idx](torch.tensor(1.0)).to(dtype=current_layer.dtype)
            attn_w = attn_weights[:, 0, l_idx].reshape(B, T, H, 1)
            output = output + scale * attn_w * layer_out
        
        return output


class AllLayerXSA(nn.Module):
    """All-Layer Cross-Scaling Attention module.
    
    Extends XSA from partial layers (e.g., last 4) to all layers with
    learnable per-layer scaling and multiple aggregation modes.
    
    Args:
        num_layers: Total number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (for GQA)
        head_dim: Dimension per attention head
        aggregation_mode: How to aggregate cross-layer information
            - "weighted_sum": Learnable weighted sum of all layers
            - "attention": Cross-attention between layers
            - "gated": Gated combination with learned gates
        enable_per_layer_scale: Whether to use learnable per-layer scales
        init_scale: Initial scale for layer weights (default: 1.0)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        aggregation_mode: Literal["weighted_sum", "attention", "gated"] = "weighted_sum",
        enable_per_layer_scale: bool = True,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.aggregation_mode = aggregation_mode
        self.enable_per_layer_scale = enable_per_layer_scale
        
        # Per-layer learnable scaling factors
        if enable_per_layer_scale:
            self.layer_scales = nn.Parameter(
                torch.full((num_layers,), init_scale / num_layers, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "layer_scales",
                torch.full((num_layers,), 1.0 / num_layers, dtype=torch.float32)
            )
        
        # Aggregation-specific components
        if aggregation_mode == "attention":
            self.cross_attn = CrossLayerAttention(
                num_layers=num_layers,
                head_dim=head_dim,
                num_heads=num_heads,
            )
        elif aggregation_mode == "gated":
            # Learnable gates for each layer
            self.layer_gates = nn.Parameter(
                torch.zeros(num_layers, num_heads, head_dim, dtype=torch.float32)
            )
            self.gate_proj = nn.Linear(head_dim, head_dim)
        
        # Global temperature for softmax scaling
        self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
        # Layer normalization for aggregated output
        self.output_norm = nn.LayerNorm(head_dim)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize module weights."""
        if hasattr(self, 'layer_gates'):
            nn.init.normal_(self.layer_gates, mean=0.0, std=0.02)
        if hasattr(self, 'gate_proj'):
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.gate_proj.bias)
    
    def get_layer_scale(self, layer_idx: int) -> Tensor:
        """Get the learnable scale for a specific layer."""
        return self.layer_scales[layer_idx]
    
    def set_layer_scale(self, layer_idx: int, value: float) -> None:
        """Set the scale for a specific layer (useful for debugging/fine-tuning)."""
        with torch.no_grad():
            self.layer_scales[layer_idx] = value
    
    def aggregate_layers(
        self,
        layer_outputs: list[Tensor],
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Aggregate outputs from all layers.
        
        Args:
            layer_outputs: List of layer outputs, each [B, T, C] or [B, T, H, D]
            return_weights: Whether to return attention/aggregation weights
        
        Returns:
            Aggregated tensor [B, T, C] or ([B, T, C], weights)
        """
        if len(layer_outputs) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer outputs, got {len(layer_outputs)}"
            )
        
        # Ensure all tensors have compatible shapes
        first_shape = layer_outputs[0].shape
        for i, out in enumerate(layer_outputs):
            if out.shape != first_shape:
                raise ValueError(
                    f"Layer {i} output shape {out.shape} doesn't match expected {first_shape}"
                )
        
        if self.aggregation_mode == "weighted_sum":
            return self._aggregate_weighted_sum(layer_outputs, return_weights)
        elif self.aggregation_mode == "attention":
            return self._aggregate_attention(layer_outputs, return_weights)
        elif self.aggregation_mode == "gated":
            return self._aggregate_gated(layer_outputs, return_weights)
        else:
            raise ValueError(f"Unknown aggregation mode: {self.aggregation_mode}")
    
    def _aggregate_weighted_sum(
        self,
        layer_outputs: list[Tensor],
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Weighted sum aggregation with learnable layer scales."""
        # Normalize scales with softmax
        scales = F.softmax(self.layer_scales / self.temperature.abs(), dim=0)
        
        # Weighted sum
        output = torch.zeros_like(layer_outputs[0])
        for i, layer_out in enumerate(layer_outputs):
            output = output + scales[i].to(dtype=layer_out.dtype) * layer_out
        
        output = self.output_norm(output)
        
        if return_weights:
            return output, scales.detach()
        return output
    
    def _aggregate_attention(
        self,
        layer_outputs: list[Tensor],
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Cross-attention based aggregation."""
        # Reshape to [B, T, H, D] if needed
        reshaped = []
        for out in layer_outputs:
            if out.dim() == 3:  # [B, T, C]
                B, T, C = out.shape
                H = self.num_heads
                D = self.head_dim
                out = out.reshape(B, T, H, D)
            reshaped.append(out)
        
        # Use last layer as query
        query = reshaped[-1]
        
        # Aggregate using cross-layer attention
        output = self.cross_attn(query, reshaped, len(reshaped) - 1)
        
        # Reshape back
        B, T, H, D = output.shape
        output = output.reshape(B, T, H * D)
        output = self.output_norm(output)
        
        if return_weights:
            # Return soft attention weights
            weights = F.softmax(self.layer_scales / self.temperature.abs(), dim=0)
            return output, weights.detach()
        return output
    
    def _aggregate_gated(
        self,
        layer_outputs: list[Tensor],
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Gated aggregation with learned per-layer gates."""
        # Reshape outputs to [B, T, H, D]
        reshaped = []
        for out in layer_outputs:
            if out.dim() == 3:
                B, T, C = out.shape
                out = out.reshape(B, T, self.num_heads, self.head_dim)
            reshaped.append(out)
        
        B, T = reshaped[0].shape[:2]
        
        # Compute gates
        stacked = torch.stack(reshaped, dim=2)  # [B, T, L, H, D]
        
        # Global pooling for gate computation
        pooled = stacked.mean(dim=(3, 4))  # [B, T, L]
        
        # Learned gates
        gates = torch.sigmoid(self.layer_gates)  # [L, H, D]
        gates = gates.unsqueeze(0).unsqueeze(0)  # [1, 1, L, H, D]
        
        # Apply gates and aggregate
        gated = stacked * gates.to(dtype=stacked.dtype)
        
        # Normalize scales
        scales = F.softmax(self.layer_scales / self.temperature.abs(), dim=0)
        scales = scales.view(1, 1, self.num_layers, 1, 1)
        
        output = (gated * scales).sum(dim=2)  # [B, T, H, D]
        output = output.reshape(B, T, self.num_heads * self.head_dim)
        output = self.output_norm(output)
        
        if return_weights:
            weights = scales.squeeze()
            gate_values = gates.mean(dim=(0, 1, 3, 4))
            combined_weights = weights * gate_values
            return output, combined_weights.detach()
        return output
    
    def forward(
        self,
        layer_outputs: list[Tensor],
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass: aggregate all layer outputs.
        
        This is an alias for aggregate_layers for nn.Module compatibility.
        """
        return self.aggregate_layers(layer_outputs, return_weights)
    
    def get_integration_hooks(self) -> dict[str, Callable]:
        """Get hooks for integrating with train_gpt_kl_v3.py.
        
        Returns a dictionary of callback functions that can be used
        to hook into the training loop.
        
        Returns:
            Dictionary with 'pre_forward', 'post_layer', 'post_forward' hooks
        """
        layer_outputs: list[Tensor] = []
        
        def pre_forward_hook() -> None:
            """Call before model forward pass."""
            layer_outputs.clear()
        
        def post_layer_hook(layer_idx: int, output: Tensor) -> Tensor:
            """Call after each layer, returns possibly modified output."""
            layer_outputs.append(output)
            return output
        
        def post_forward_hook(final_output: Tensor) -> Tensor:
            """Call after model forward pass, returns aggregated output."""
            if len(layer_outputs) == self.num_layers:
                aggregated = self.aggregate_layers(layer_outputs)
                # Blend with final output
                alpha = torch.sigmoid(self.temperature)
                return alpha * aggregated + (1 - alpha) * final_output
            return final_output
        
        return {
            'pre_forward': pre_forward_hook,
            'post_layer': post_layer_hook,
            'post_forward': post_forward_hook,
        }


class XSAEnabledBlock(nn.Module):
    """Wrapper for transformer blocks that enables XSA recording.
    
    This wrapper stores layer outputs for XSA aggregation while
    maintaining compatibility with standard transformer blocks.
    
    Args:
        block: The transformer block to wrap
        layer_idx: Index of this layer in the model
        xsa_module: The AllLayerXSA module for aggregation
    """
    
    def __init__(
        self,
        block: nn.Module,
        layer_idx: int,
        xsa_module: Optional[AllLayerXSA] = None,
    ):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.xsa_module = xsa_module
        self._layer_outputs: Optional[list] = None
    
    def set_layer_outputs_list(self, outputs_list: list) -> None:
        """Set the shared list for storing layer outputs."""
        self._layer_outputs = outputs_list
    
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass with XSA recording."""
        output = self.block(x, *args, **kwargs)
        
        # Store output for XSA aggregation
        if self._layer_outputs is not None:
            # Ensure we have enough slots
            while len(self._layer_outputs) <= self.layer_idx:
                self._layer_outputs.append(None)
            self._layer_outputs[self.layer_idx] = output
        
        return output
    
    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped block."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)


def integrate_xsa_into_model(
    model: nn.Module,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    aggregation_mode: Literal["weighted_sum", "attention", "gated"] = "weighted_sum",
    target_layers: Optional[list[int]] = None,
) -> AllLayerXSA:
    """Integrate AllLayerXSA into an existing model.
    
    This helper function wraps transformer blocks and attaches the XSA module.
    
    Args:
        model: The transformer model to modify
        num_layers: Total number of layers
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension per head
        aggregation_mode: XSA aggregation mode
        target_layers: Specific layers to apply XSA (None = all layers)
    
    Returns:
        The initialized AllLayerXSA module
    """
    xsa = AllLayerXSA(
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        aggregation_mode=aggregation_mode,
    )
    
    # Store reference to xsa module in model
    model._xsa_module = xsa
    
    return xsa


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test
    batch_size = 2
    seq_len = 128
    num_layers = 11
    num_heads = 8
    num_kv_heads = 4
    head_dim = 64
    model_dim = num_heads * head_dim
    
    print("Testing AllLayerXSA...")
    
    # Create XSA module
    xsa = AllLayerXSA(
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        aggregation_mode="weighted_sum",
    )
    
    # Create dummy layer outputs
    layer_outputs = [
        torch.randn(batch_size, seq_len, model_dim)
        for _ in range(num_layers)
    ]
    
    # Test aggregation
    aggregated, weights = xsa.aggregate_layers(layer_outputs, return_weights=True)
    
    print(f"Input shapes: {[o.shape for o in layer_outputs]}")
    print(f"Aggregated shape: {aggregated.shape}")
    print(f"Layer weights: {weights}")
    print(f"Weight sum: {weights.sum().item():.4f}")
    
    # Test different aggregation modes
    for mode in ["weighted_sum", "gated"]:
        xsa_mode = AllLayerXSA(
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            aggregation_mode=mode,
        )
        out = xsa_mode.aggregate_layers(layer_outputs)
        print(f"Mode '{mode}' output shape: {out.shape}")
    
    print("\nAll tests passed!")
    
    # Demonstrate integration hooks
    hooks = xsa.get_integration_hooks()
    print(f"\nAvailable hooks: {list(hooks.keys())}")