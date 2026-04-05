"""Tests for fractal_compressor.py — Stage 1, 2, 3 modules."""

import math

import torch
import torch.nn.functional as F

from fractal_compressor import (
    ByteAssembler,
    FoldedRecurrentBlock,
    FractalCompressor,
    SequenceFolder,
    SharedProjector,
    SpectralLinear,
    _build_dct_basis,
)


def test_dct_basis_orthonormality():
    """DCT basis should be approximately orthonormal: B @ B^T ≈ I."""
    for n in [8, 16, 32, 64]:
        B = _build_dct_basis(n)
        I_approx = B @ B.T
        I_true = torch.eye(n)
        assert torch.allclose(I_approx, I_true, atol=1e-5), (
            f"DCT basis not orthonormal for n={n}, max error={torch.max(torch.abs(I_approx - I_true)):.6f}"
        )


def test_spectral_linear_shapes():
    """SpectralLinear should produce correct output shapes."""
    for in_f, out_f in [(64, 128), (128, 64), (32, 32)]:
        layer = SpectralLinear(in_f, out_f, rank_frac=0.3)
        x = torch.randn(2, 10, in_f)
        y = layer(x)
        assert y.shape == (2, 10, out_f), f"Expected (2,10,{out_f}), got {y.shape}"


def test_spectral_linear_parameter_savings():
    """SpectralLinear should have fewer learnable params than a full nn.Linear."""
    in_f, out_f = 256, 512
    spectral = SpectralLinear(in_f, out_f, rank_frac=0.3)
    full = torch.nn.Linear(in_f, out_f, bias=False)
    spectral_params = sum(p.numel() for p in spectral.parameters())
    full_params = sum(p.numel() for p in full.parameters())
    assert spectral_params < full_params, (
        f"SpectralLinear should have fewer params: {spectral_params} vs {full_params}"
    )


def test_spectral_linear_gradient_flow():
    """Gradients should flow through the DCT reconstruction."""
    layer = SpectralLinear(32, 64, rank_frac=0.3)
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert layer.coeffs.grad is not None, "No gradient on coefficients"
    assert layer.coeffs.grad.abs().sum() > 0, "Zero gradient on coefficients"
    assert x.grad is not None, "No gradient on input"


def test_spectral_linear_with_bias():
    """SpectralLinear with bias should work."""
    layer = SpectralLinear(32, 64, rank_frac=0.3, bias=True)
    x = torch.randn(2, 5, 32)
    y = layer(x)
    assert y.shape == (2, 5, 64)
    assert layer.bias is not None


def test_shared_projector_shapes():
    """SharedProjector should produce correct shapes from both paths."""
    proj = SharedProjector(64, 128, rank_frac=0.3)
    x = torch.randn(2, 10, 64)
    a = proj.forward_a(x)
    b = proj.forward_b(x)
    assert a.shape == (2, 10, 128)
    assert b.shape == (2, 10, 128)


def test_shared_projector_shared_weights():
    """Both projections should share the same base weights but differ by shift."""
    proj = SharedProjector(32, 64, rank_frac=0.3)
    x = torch.randn(1, 5, 32)
    a = proj.forward_a(x)
    b = proj.forward_b(x)
    # With zero shifts initialized, both should be identical
    assert torch.allclose(a, b, atol=1e-6), "With zero shifts, outputs should match"
    # After modifying one shift, they should differ
    with torch.no_grad():
        proj.shift_a.fill_(1.0)
    a2 = proj.forward_a(x)
    b2 = proj.forward_b(x)
    assert not torch.allclose(a2, b2, atol=1e-3), "With different shifts, outputs should differ"


def test_sequence_folder_even_length():
    """SequenceFolder should halve even-length sequences."""
    folder = SequenceFolder(dim=32)
    x = torch.randn(2, 10, 32)
    y = folder(x)
    assert y.shape == (2, 5, 32), f"Expected (2,5,32), got {y.shape}"


def test_sequence_folder_odd_length():
    """SequenceFolder should handle odd-length sequences: ceil(L/2)."""
    folder = SequenceFolder(dim=32)
    x = torch.randn(2, 11, 32)
    y = folder(x)
    assert y.shape == (2, 6, 32), f"Expected (2,6,32), got {y.shape}"


def test_sequence_folder_length_one():
    """SequenceFolder should pass through length-1 sequences unchanged."""
    folder = SequenceFolder(dim=32)
    x = torch.randn(2, 1, 32)
    y = folder(x)
    assert y.shape == (2, 1, 32)
    assert torch.equal(x, y)


def test_sequence_folder_gradient_flow():
    """Gradients should flow through the SequenceFolder."""
    folder = SequenceFolder(dim=16)
    x = torch.randn(2, 8, 16, requires_grad=True)
    y = folder(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_folded_recurrent_block_no_fold():
    """FoldedRecurrentBlock should not fold on iteration 0."""
    block = FoldedRecurrentBlock(dim=64, num_heads=4, num_kv_heads=2, mlp_mult=2, fold_every=3)
    x = torch.randn(2, 16, 64)
    y = block(x, iteration=0)
    assert y.shape == (2, 16, 64), "Iteration 0 should not fold"


def test_folded_recurrent_block_fold_trigger():
    """FoldedRecurrentBlock should fold on iteration == fold_every."""
    block = FoldedRecurrentBlock(dim=64, num_heads=4, num_kv_heads=2, mlp_mult=2, fold_every=3)
    x = torch.randn(2, 16, 64)
    y = block(x, iteration=3)
    assert y.shape[1] == 8, f"Expected sequence length 8 after fold, got {y.shape[1]}"


def test_folded_recurrent_block_no_fold_other_iterations():
    """FoldedRecurrentBlock should not fold on non-multiple iterations."""
    block = FoldedRecurrentBlock(dim=64, num_heads=4, num_kv_heads=2, mlp_mult=2, fold_every=3)
    x = torch.randn(2, 16, 64)
    y = block(x, iteration=1)
    assert y.shape == (2, 16, 64)
    y = block(x, iteration=2)
    assert y.shape == (2, 16, 64)


def test_byte_assembler_shapes():
    """ByteAssembler should map byte IDs to model_dim vectors."""
    asm = ByteAssembler(byte_dim=32, model_dim=128)
    byte_ids = torch.randint(0, 256, (2, 20))
    y = asm(byte_ids)
    assert y.shape == (2, 20, 128)


def test_byte_assembler_gradient():
    """ByteAssembler should be differentiable."""
    asm = ByteAssembler(byte_dim=16, model_dim=64)
    byte_ids = torch.randint(0, 256, (2, 10))
    y = asm(byte_ids)
    loss = y.sum()
    loss.backward()
    assert asm.fc1.weight.grad is not None


def test_fractal_compressor_forward():
    """FractalCompressor should produce a scalar loss."""
    model = FractalCompressor(
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        num_iterations=2,
        fold_every=3,  # No folding in 2 iterations
        max_seq_len=32,
        byte_dim=16,
        vocab_size=256,
    )
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    loss = model(x, y)
    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"


def test_fractal_compressor_with_folding():
    """FractalCompressor with folding should still produce valid loss."""
    model = FractalCompressor(
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        num_iterations=4,
        fold_every=2,   # Fold at iterations 2 and 4
        max_seq_len=64,
        byte_dim=16,
        vocab_size=256,
    )
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    loss = model(x, y)
    assert loss.shape == ()
    assert not torch.isnan(loss), "Loss is NaN"


def test_fractal_compressor_backward():
    """FractalCompressor should support full backward pass."""
    model = FractalCompressor(
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        num_iterations=2,
        fold_every=5,
        max_seq_len=16,
        byte_dim=8,
        vocab_size=256,
    )
    x = torch.randint(0, 256, (1, 8))
    y = torch.randint(0, 256, (1, 8))
    loss = model(x, y)
    loss.backward()
    # Check that key parameters received gradients
    assert model.assembler.fc1.weight.grad is not None
    assert model.recurrent_block.c_q.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_fractal_compressor_forward_logits():
    """forward_logits should return logits tensor."""
    model = FractalCompressor(
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        num_iterations=2,
        fold_every=5,
        max_seq_len=16,
        byte_dim=8,
        vocab_size=256,
    )
    x = torch.randint(0, 256, (1, 8))
    logits = model.forward_logits(x)
    assert logits.ndim == 3
    assert logits.shape[0] == 1
    assert logits.shape[2] == 256


def test_fractal_compressor_parameter_count():
    """FractalCompressor parameters should be well under 16M for a small config."""
    model = FractalCompressor(
        model_dim=128,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        num_iterations=4,
        fold_every=3,
        max_seq_len=256,
        byte_dim=32,
        vocab_size=1024,
    )
    n_params = sum(p.numel() for p in model.parameters())
    # This small config should be well under 16M parameters
    assert n_params < 16_000_000, f"Too many params: {n_params}"
    print(f"FractalCompressor param count (small): {n_params:,}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test_fn in tests:
        print(f"Running {test_fn.__name__}... ", end="", flush=True)
        test_fn()
        print("✓")
    print(f"\nAll {len(tests)} tests passed!")
