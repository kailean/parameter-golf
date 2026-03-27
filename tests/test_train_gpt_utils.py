"""Tests for train_gpt.py utility functions.

All tests here run on CPU only — no CUDA required. They validate the
math-critical and data-pipeline functions that sit beneath the training loop:
orthogonalization, quantization, rotary embeddings, model forward pass, and
token-shard I/O.

Run with:
    pytest tests/test_train_gpt_utils.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import train_gpt
from train_gpt import (
    DATAFILE_MAGIC,
    CausalSelfAttention,
    GPT,
    INT8_KEEP_FLOAT_STORE_DTYPE,
    TokenStream,
    apply_rotary_emb,
    dequantize_state_dict_int8,
    keep_float_tensor,
    load_data_shard,
    quantize_float_tensor,
    quantize_state_dict_int8,
    tensor_nbytes,
    zeropower_via_newtonschulz5,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_shard(path: Path, tokens: np.ndarray) -> None:
    """Write a minimal valid binary shard file."""
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = 1   # DATAFILE_VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2").tobytes())


# ---------------------------------------------------------------------------
# zeropower_via_newtonschulz5
# ---------------------------------------------------------------------------

class TestZeropowerViaNewtonSchulz5:
    def test_square_matrix_output_is_near_orthonormal(self):
        torch.manual_seed(42)
        G = torch.randn(8, 8)
        X = zeropower_via_newtonschulz5(G, steps=10)
        prod = X.float() @ X.float().T
        torch.testing.assert_close(prod, torch.eye(8), atol=0.05, rtol=0)

    def test_wide_matrix_rows_are_orthonormal(self):
        """For a wide matrix (rows < cols), X @ X.T should be near identity."""
        torch.manual_seed(0)
        G = torch.randn(4, 8)
        X = zeropower_via_newtonschulz5(G, steps=10)
        prod = X.float() @ X.float().T
        torch.testing.assert_close(prod, torch.eye(4), atol=0.05, rtol=0)

    def test_tall_matrix_cols_are_orthonormal(self):
        """For a tall matrix (rows > cols), X.T @ X should be near identity."""
        torch.manual_seed(1)
        G = torch.randn(8, 4)
        X = zeropower_via_newtonschulz5(G, steps=10)
        prod = X.float().T @ X.float()
        torch.testing.assert_close(prod, torch.eye(4), atol=0.05, rtol=0)

    def test_output_shape_matches_input(self):
        G = torch.randn(6, 3)
        X = zeropower_via_newtonschulz5(G, steps=5)
        assert X.shape == G.shape

    def test_output_dtype_is_bfloat16(self):
        G = torch.randn(4, 4)
        X = zeropower_via_newtonschulz5(G, steps=5)
        assert X.dtype == torch.bfloat16

    def test_zero_gradient_does_not_raise(self):
        """A zero matrix should not crash (eps guard prevents div-by-zero)."""
        G = torch.zeros(4, 4)
        X = zeropower_via_newtonschulz5(G, steps=5)
        assert X.shape == (4, 4)
        assert not torch.isnan(X).any()


# ---------------------------------------------------------------------------
# quantize_float_tensor
# ---------------------------------------------------------------------------

class TestQuantizeFloatTensor:
    def test_2d_tensor_produces_per_row_scale(self):
        t = torch.randn(4, 8)
        q, s = quantize_float_tensor(t)
        assert q.dtype == torch.int8
        assert q.shape == t.shape
        assert s.shape == (4,)

    def test_1d_tensor_produces_scalar_scale(self):
        t = torch.randn(16)
        q, s = quantize_float_tensor(t)
        assert q.dtype == torch.int8
        assert q.shape == t.shape
        assert s.ndim == 0

    def test_quantized_values_within_int8_range(self):
        t = torch.randn(8, 8) * 100
        q, _ = quantize_float_tensor(t)
        assert int(q.min()) >= -127
        assert int(q.max()) <= 127

    def test_dequantize_is_approximate_inverse(self):
        torch.manual_seed(0)
        t = torch.randn(16, 16)
        q, s = quantize_float_tensor(t)
        t_hat = q.float() * s[:, None]
        rel_err = (t_hat - t).abs() / (t.abs().mean() + 1e-6)
        assert rel_err.mean() < 0.05

    def test_empty_2d_tensor(self):
        t = torch.empty(0, 8)
        q, s = quantize_float_tensor(t)
        assert q.shape == (0, 8)
        assert s.shape == (0,)


# ---------------------------------------------------------------------------
# quantize_state_dict_int8 / dequantize_state_dict_int8
# ---------------------------------------------------------------------------

class TestQuantizeStateDictRoundtrip:
    def _make_state_dict(self):
        return {
            "blocks.0.attn.c_q.weight": torch.randn(128, 128),   # large matrix → int8
            "tok_emb.weight": torch.randn(256, 32),               # large matrix → int8
            "blocks.0.attn_scale": torch.randn(32),               # small → passthrough
        }

    def test_roundtrip_preserves_all_keys(self):
        sd = self._make_state_dict()
        obj, _ = quantize_state_dict_int8(sd)
        deq = dequantize_state_dict_int8(obj)
        assert set(deq.keys()) == set(sd.keys())

    def test_roundtrip_preserves_shape(self):
        sd = self._make_state_dict()
        obj, _ = quantize_state_dict_int8(sd)
        deq = dequantize_state_dict_int8(obj)
        for name in sd:
            assert deq[name].shape == sd[name].shape

    def test_roundtrip_values_approximately_correct(self):
        torch.manual_seed(0)
        sd = {"w": torch.randn(64, 64)}
        obj, _ = quantize_state_dict_int8(sd)
        deq = dequantize_state_dict_int8(obj)
        rel_err = (deq["w"] - sd["w"]).abs() / (sd["w"].abs() + 1e-6)
        assert rel_err.mean() < 0.05

    def test_non_float_tensor_passes_through_unchanged(self):
        sd = {"mask": torch.tensor([1, 2, 3], dtype=torch.int32)}
        obj, _ = quantize_state_dict_int8(sd)
        deq = dequantize_state_dict_int8(obj)
        assert deq["mask"].dtype == torch.int32
        torch.testing.assert_close(deq["mask"], sd["mask"])

    def test_stats_keys_are_present(self):
        sd = {"w": torch.randn(64, 64)}
        _, stats = quantize_state_dict_int8(sd)
        for key in ("param_count", "num_tensors", "baseline_tensor_bytes", "int8_payload_bytes"):
            assert key in stats, f"Missing stats key: {key}"

    def test_int8_payload_smaller_than_fp32_baseline(self):
        sd = {"w": torch.randn(256, 256)}
        _, stats = quantize_state_dict_int8(sd)
        assert stats["int8_payload_bytes"] < stats["baseline_tensor_bytes"]

    def test_small_tensor_is_not_quantized(self):
        """Tensors with ≤ INT8_KEEP_FLOAT_MAX_NUMEL elements go through passthrough."""
        small = torch.randn(32)   # well below 65536
        obj, _ = quantize_state_dict_int8({"s": small})
        assert "s" not in obj["quantized"]
        assert "s" in obj["passthrough"]

    def test_control_pattern_tensor_keeps_full_precision(self):
        """Tensors matching CONTROL_TENSOR_NAME_PATTERNS are kept as float32."""
        sd = {"attn_scale": torch.randn(512)}
        obj, _ = quantize_state_dict_int8(sd)
        deq = dequantize_state_dict_int8(obj)
        assert deq["attn_scale"].dtype == torch.float32

    def test_quant_format_key_present(self):
        obj, _ = quantize_state_dict_int8({"w": torch.randn(64, 64)})
        assert obj.get("__quant_format__") == "int8_clean_per_row_v1"


# ---------------------------------------------------------------------------
# tensor_nbytes
# ---------------------------------------------------------------------------

class TestTensorNbytes:
    def test_float32(self):
        t = torch.zeros(4, 4, dtype=torch.float32)
        assert tensor_nbytes(t) == 4 * 4 * 4  # 4 bytes per element

    def test_int8(self):
        t = torch.zeros(8, dtype=torch.int8)
        assert tensor_nbytes(t) == 8

    def test_float16(self):
        t = torch.zeros(10, dtype=torch.float16)
        assert tensor_nbytes(t) == 20

    def test_scalar_tensor(self):
        t = torch.tensor(1.0, dtype=torch.float32)
        assert tensor_nbytes(t) == 4


# ---------------------------------------------------------------------------
# keep_float_tensor
# ---------------------------------------------------------------------------

class TestKeepFloatTensor:
    def test_control_pattern_preserves_float32(self):
        t = torch.randn(4, dtype=torch.float32)
        orig_dtypes: dict = {}
        result = keep_float_tensor("attn_scale", t, orig_dtypes)
        assert result.dtype == torch.float32

    def test_bfloat16_is_downcast_to_fp16(self):
        t = torch.randn(4, dtype=torch.bfloat16)
        orig_dtypes: dict = {}
        result = keep_float_tensor("weight", t, orig_dtypes)
        assert result.dtype == INT8_KEEP_FLOAT_STORE_DTYPE

    def test_original_dtype_is_recorded(self):
        t = torch.randn(4, dtype=torch.bfloat16)
        orig_dtypes: dict = {}
        keep_float_tensor("weight", t, orig_dtypes)
        assert "weight" in orig_dtypes
        assert orig_dtypes["weight"] == "bfloat16"

    def test_fp32_is_downcast_and_dtype_recorded(self):
        t = torch.randn(4, dtype=torch.float32)
        orig_dtypes: dict = {}
        result = keep_float_tensor("some_weight", t, orig_dtypes)
        assert result.dtype == INT8_KEEP_FLOAT_STORE_DTYPE
        assert orig_dtypes.get("some_weight") == "float32"

    def test_mlp_scale_control_pattern_keeps_float32(self):
        t = torch.randn(4, dtype=torch.bfloat16)
        orig_dtypes: dict = {}
        result = keep_float_tensor("mlp_scale", t, orig_dtypes)
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# apply_rotary_emb
# ---------------------------------------------------------------------------

class TestApplyRotaryEmb:
    def test_output_shape_matches_input(self):
        x = torch.randn(2, 4, 8, 16)
        cos = torch.ones(1, 1, 8, 8)
        sin = torch.zeros(1, 1, 8, 8)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_identity_when_sin_zero_cos_one(self):
        """cos=1, sin=0 is the identity rotation."""
        x = torch.randn(1, 1, 4, 8)
        half = 4
        cos = torch.ones(1, 1, 4, half)
        sin = torch.zeros(1, 1, 4, half)
        out = apply_rotary_emb(x, cos, sin)
        torch.testing.assert_close(out, x)

    def test_output_dtype_matches_input(self):
        x = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        cos = torch.ones(1, 1, 4, 4, dtype=torch.float16)
        sin = torch.zeros(1, 1, 4, 4, dtype=torch.float16)
        out = apply_rotary_emb(x, cos, sin)
        assert out.dtype == torch.float16

    def test_rotation_is_norm_preserving(self):
        """Rotation should not change the L2 norm of each token vector."""
        torch.manual_seed(7)
        x = torch.randn(1, 1, 4, 8)
        angle = 0.5
        # Build cos/sin for a uniform rotation
        cos = torch.full((1, 1, 4, 4), fill_value=float(torch.cos(torch.tensor(angle))))
        sin = torch.full((1, 1, 4, 4), fill_value=float(torch.sin(torch.tensor(angle))))
        out = apply_rotary_emb(x, cos, sin)
        torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# CausalSelfAttention validation
# ---------------------------------------------------------------------------

class TestCausalSelfAttentionValidation:
    def test_dim_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            CausalSelfAttention(dim=33, num_heads=8, num_kv_heads=4,
                                rope_base=10000.0, qk_gain_init=1.5)

    def test_heads_not_divisible_by_kv_heads_raises(self):
        with pytest.raises(ValueError, match="divisible by num_kv_heads"):
            CausalSelfAttention(dim=32, num_heads=8, num_kv_heads=3,
                                rope_base=10000.0, qk_gain_init=1.5)

    def test_odd_head_dim_raises(self):
        # dim=24, num_heads=8 → head_dim=3, which is odd
        with pytest.raises(ValueError, match="even"):
            CausalSelfAttention(dim=24, num_heads=8, num_kv_heads=8,
                                rope_base=10000.0, qk_gain_init=1.5)

    def test_valid_config_constructs_without_error(self):
        attn = CausalSelfAttention(dim=32, num_heads=4, num_kv_heads=2,
                                   rope_base=10000.0, qk_gain_init=1.5)
        assert attn.head_dim == 8


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    """A minimal GPT that runs quickly on CPU."""
    return GPT(
        vocab_size=64,
        num_layers=2,
        model_dim=32,
        num_heads=2,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )


class TestGPTModel:
    def test_forward_returns_scalar_loss(self, tiny_model):
        x = torch.randint(0, 64, (2, 8))
        y = torch.randint(0, 64, (2, 8))
        loss = tiny_model(x, y)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_loss_finite_at_init(self, tiny_model):
        x = torch.randint(0, 64, (2, 8))
        y = torch.randint(0, 64, (2, 8))
        loss = tiny_model(x, y)
        assert torch.isfinite(loss)

    def test_loss_near_log_vocab_size_at_random_init(self):
        """At random init, cross-entropy should be close to log(vocab_size)."""
        torch.manual_seed(42)
        model = GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.02,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        x = torch.randint(0, 64, (4, 16))
        y = torch.randint(0, 64, (4, 16))
        loss = model(x, y)
        expected = float(np.log(64))  # ~4.16
        assert abs(loss.item() - expected) < 2.0

    def test_invalid_logit_softcap_raises(self):
        with pytest.raises(ValueError, match="logit_softcap"):
            GPT(vocab_size=64, num_layers=2, model_dim=32, num_heads=2,
                num_kv_heads=2, mlp_mult=2, tie_embeddings=True,
                tied_embed_init_std=0.02, logit_softcap=0.0,
                rope_base=10000.0, qk_gain_init=1.5)

    def test_untied_embeddings_has_lm_head(self):
        model = GPT(
            vocab_size=64, num_layers=2, model_dim=32, num_heads=2,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=False,
            tied_embed_init_std=0.02, logit_softcap=30.0,
            rope_base=10000.0, qk_gain_init=1.5,
        )
        assert model.lm_head is not None

    def test_untied_model_forward(self):
        model = GPT(
            vocab_size=64, num_layers=2, model_dim=32, num_heads=2,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=False,
            tied_embed_init_std=0.02, logit_softcap=30.0,
            rope_base=10000.0, qk_gain_init=1.5,
        )
        loss = model(torch.randint(0, 64, (1, 4)), torch.randint(0, 64, (1, 4)))
        assert torch.isfinite(loss)

    def test_tied_embeddings_has_no_lm_head(self, tiny_model):
        assert tiny_model.lm_head is None

    def test_skip_weights_count(self):
        """skip_weights size should be min(encoder_layers, decoder_layers)."""
        model = GPT(
            vocab_size=64, num_layers=4, model_dim=32, num_heads=2, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.02,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        assert model.skip_weights.shape[0] == 2  # min(2 encoder, 2 decoder)

    def test_odd_num_layers_split(self):
        """With 3 layers: 1 encoder, 2 decoder → 1 skip weight."""
        model = GPT(
            vocab_size=64, num_layers=3, model_dim=32, num_heads=2, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.02,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        assert model.num_encoder_layers == 1
        assert model.num_decoder_layers == 2
        assert model.skip_weights.shape[0] == 1


# ---------------------------------------------------------------------------
# load_data_shard / TokenStream
# ---------------------------------------------------------------------------

class TestLoadDataShard:
    def test_loads_valid_shard(self, tmp_path):
        tokens = np.arange(10, dtype=np.uint16)
        path = tmp_path / "shard.bin"
        _write_shard(path, tokens)
        result = load_data_shard(path)
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == list(range(10))

    def test_rejects_wrong_magic(self, tmp_path):
        path = tmp_path / "bad.bin"
        header = np.zeros(256, dtype="<i4")
        header[0] = 99999  # wrong magic
        header[1] = 1
        header[2] = 0
        with open(path, "wb") as f:
            f.write(header.tobytes())
        with pytest.raises(ValueError, match="header"):
            load_data_shard(path)

    def test_rejects_size_mismatch(self, tmp_path):
        tokens = np.arange(10, dtype=np.uint16)
        path = tmp_path / "truncated.bin"
        _write_shard(path, tokens)
        # Truncate the file by removing some token bytes
        data = path.read_bytes()
        path.write_bytes(data[:-4])
        with pytest.raises(ValueError, match="[Ss]ize|[Mm]ismatch|[Hh]eader"):
            load_data_shard(path)

    def test_empty_shard(self, tmp_path):
        path = tmp_path / "empty.bin"
        _write_shard(path, np.array([], dtype=np.uint16))
        result = load_data_shard(path)
        assert result.numel() == 0


class TestTokenStream:
    def test_take_initial_tokens(self, tmp_path):
        tokens = np.arange(20, dtype=np.uint16)
        shard = tmp_path / "fineweb_train_000000.bin"
        _write_shard(shard, tokens)
        stream = TokenStream(str(tmp_path / "fineweb_train_*.bin"))
        result = stream.take(5)
        assert result.tolist() == [0, 1, 2, 3, 4]

    def test_take_advances_position(self, tmp_path):
        tokens = np.arange(20, dtype=np.uint16)
        shard = tmp_path / "fineweb_train_000000.bin"
        _write_shard(shard, tokens)
        stream = TokenStream(str(tmp_path / "fineweb_train_*.bin"))
        stream.take(5)
        result = stream.take(3)
        assert result.tolist() == [5, 6, 7]

    def test_take_wraps_around_single_shard(self, tmp_path):
        tokens = np.arange(5, dtype=np.uint16)
        shard = tmp_path / "fineweb_train_000000.bin"
        _write_shard(shard, tokens)
        stream = TokenStream(str(tmp_path / "fineweb_train_*.bin"))
        stream.take(5)  # exhaust the shard
        result = stream.take(3)  # should wrap back to start
        assert result.tolist() == [0, 1, 2]

    def test_take_spans_multiple_shards(self, tmp_path):
        for i in range(2):
            _write_shard(
                tmp_path / f"fineweb_train_{i:06d}.bin",
                np.array([i * 10 + j for j in range(5)], dtype=np.uint16),
            )
        stream = TokenStream(str(tmp_path / "fineweb_train_*.bin"))
        # Take 8 tokens — first 5 from shard 0, then 3 from shard 1
        result = stream.take(8)
        assert result.tolist() == [0, 1, 2, 3, 4, 10, 11, 12]

    def test_no_files_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TokenStream(str(tmp_path / "nonexistent_*.bin"))

    def test_returned_tensor_length_matches_requested(self, tmp_path):
        tokens = np.arange(50, dtype=np.uint16)
        _write_shard(tmp_path / "fineweb_train_000000.bin", tokens)
        stream = TokenStream(str(tmp_path / "fineweb_train_*.bin"))
        for n in [1, 7, 13, 25]:
            assert len(stream.take(n)) == n
