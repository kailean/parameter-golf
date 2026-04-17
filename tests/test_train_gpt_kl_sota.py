import importlib
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_NAME = "train_gpt_kl"
SOTA_ENV_KEYS = (
    "DATA_PATH",
    "TOKENIZER_PATH",
    "VOCAB_SIZE",
    "NUM_LAYERS",
    "LAYERS",
    "MODEL_DIM",
    "DIM",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "MLP_MULT",
    "BIGRAM_HASH_SIZE",
    "XSA_LAST_N",
    "XSA_LAYERS",
    "QK_GAIN_INIT",
    "LEAKY_RELU_SLOPE",
    "EMBED_BITS",
    "MATRIX_CLIP_SIGMAS",
    "EMBED_CLIP_SIGMAS",
    "GPTQ_CALIBRATION_BATCHES",
    "BYTE_SHUFFLE_STRIDE",
    "VAL_LOSS_EVERY",
    "WARMDOWN_FRAC",
    "LATE_QAT_THRESHOLD",
    "TTT_RANK",
    "TTT_LR",
    "TTT_CHUNK",
    "TTT_BATCH",
    "TTT_STEPS",
    "TTT_PHASES",
    "TTT_BETA1",
    "TTT_BETA2",
    "TTT_WEIGHT_DECAY",
    "MUON_WEIGHT_DECAY",
    "EMBED_WEIGHT_DECAY",
    "EMBED_WD",
)


@contextmanager
def isolated_sota_env(overrides=None):
    saved = {key: os.environ.get(key) for key in SOTA_ENV_KEYS}
    try:
        for key in SOTA_ENV_KEYS:
            os.environ.pop(key, None)
        if overrides:
            os.environ.update(overrides)
        sys.modules.pop(MODULE_NAME, None)
        yield
    finally:
        sys.modules.pop(MODULE_NAME, None)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_module():
    return importlib.import_module(MODULE_NAME)


class TrainGptKlSotaDefaultsTest(unittest.TestCase):
    def test_sp8192_defaults_disable_bigram_hash(self):
        with isolated_sota_env():
            mod = load_module()
            args = mod.Hyperparameters()
            self.assertTrue(args.data_path.endswith("fineweb10B_sp8192"))
            self.assertTrue(args.tokenizer_path.endswith("fineweb_8192_bpe.model"))
            self.assertEqual(args.vocab_size, 8192)
            self.assertEqual(args.bigram_hash_size, 0)
            self.assertEqual(args.num_layers, 13)
            self.assertEqual(args.model_dim, 512)
            self.assertEqual(args.mlp_mult, 4)
            self.assertEqual(args.num_heads, 8)
            self.assertEqual(args.num_kv_heads, 4)

    def test_ttt_targets_and_chunk_schedule_match_sota(self):
        with isolated_sota_env():
            mod = load_module()
            model = mod.GPT(
                vocab_size=64,
                num_layers=2,
                model_dim=32,
                num_heads=4,
                num_kv_heads=2,
                mlp_mult=2,
                tie_embeddings=True,
                tied_embed_init_std=0.005,
                logit_softcap=30.0,
                rope_base=10000.0,
                qk_gain_init=5.0,
                bigram_hash_size=0,
                use_ortho_init=False,
                smear_enabled=True,
                xsa_last_n=2,
                rope_dims=16,
                ln_scale_enabled=True,
                depth_recurrence=False,
                recurrence_layers=[],
                recurrence_loops=1,
                parallel_residuals=True,
                parallel_res_start=1,
                parallel_final_lane_mean=True,
                leaky_slope=0.5,
            )
            target_names = [name for name, _module in mod.collect_ttt_target_modules(model)]
            self.assertIn("blocks.0.attn.c_q", target_names)
            self.assertIn("blocks.0.attn.c_k", target_names)
            self.assertIn("blocks.0.attn.proj", target_names)
            self.assertIn("blocks.0.mlp.fc", target_names)
            self.assertIn("blocks.0.mlp.proj", target_names)
            self.assertFalse(any(name.endswith(".attn.c_v") for name in target_names))

            chunk_schedule = mod.build_ttt_phase_chunk_schedule(seq_len=96, chunk_size=32, phases=3)
            self.assertEqual(len(chunk_schedule), 9)
            self.assertEqual(chunk_schedule[0], (0, 0, 32))
            self.assertEqual(chunk_schedule[3], (1, 0, 32))
            self.assertEqual(chunk_schedule[8], (2, 64, 96))

    def test_sota_config_summary_reports_budget(self):
        with isolated_sota_env():
            mod = load_module()
            args = mod.Hyperparameters()
            summary = mod.build_sota_config_summary(args)
            self.assertEqual(summary["vocab_size"], 8192)
            self.assertEqual(summary["bigram_hash_size"], 0)
            self.assertEqual(summary["xsa_layers"], 13)
            self.assertGreater(summary["param_count"], 40_000_000)
            self.assertLess(summary["estimated_int6_brotli_bytes"], 16 * 1024 * 1024)
            self.assertGreater(summary["submission_margin_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
