"""Tests for data/cached_challenge_fineweb.py utility functions.

Covers pure path-routing and argument-parsing logic that requires no network
access and no local dataset files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import data.cached_challenge_fineweb as mod
from data.cached_challenge_fineweb import (
    artifact_paths_for_tokenizer,
    dataset_dir_for_variant,
)


# ---------------------------------------------------------------------------
# dataset_dir_for_variant
# ---------------------------------------------------------------------------

class TestDatasetDirForVariant:
    def test_byte260(self):
        assert dataset_dir_for_variant("byte260") == "fineweb10B_byte260"

    def test_sp1024(self):
        assert dataset_dir_for_variant("sp1024") == "fineweb10B_sp1024"

    def test_sp4096(self):
        assert dataset_dir_for_variant("sp4096") == "fineweb10B_sp4096"

    def test_sp_with_large_vocab(self):
        assert dataset_dir_for_variant("sp32000") == "fineweb10B_sp32000"

    def test_unsupported_variant_raises(self):
        with pytest.raises(ValueError, match="unsupported variant"):
            dataset_dir_for_variant("gpt2")

    def test_sp_non_numeric_raises(self):
        with pytest.raises(ValueError, match="unsupported variant"):
            dataset_dir_for_variant("spNOTNUMBER")

    def test_sp_empty_number_raises(self):
        with pytest.raises(ValueError, match="unsupported variant"):
            dataset_dir_for_variant("sp")

    def test_error_message_includes_variant_name(self):
        with pytest.raises(ValueError, match="unknown_variant"):
            dataset_dir_for_variant("unknown_variant")


# ---------------------------------------------------------------------------
# local_path_for_remote
# ---------------------------------------------------------------------------

class TestLocalPathForRemote:
    def test_datasets_prefix_maps_under_datasets_dir(self):
        result = mod.local_path_for_remote("datasets/fineweb10B_sp1024/shard.bin")
        assert result.name == "shard.bin"
        assert "fineweb10B_sp1024" in str(result)

    def test_tokenizers_prefix_maps_under_tokenizers_dir(self):
        result = mod.local_path_for_remote("tokenizers/fineweb_1024_bpe.model")
        assert result.name == "fineweb_1024_bpe.model"
        assert "tokenizers" in str(result)

    def test_remote_root_prefix_is_stripped(self, monkeypatch):
        monkeypatch.setattr(mod, "REMOTE_ROOT_PREFIX", "myprefix")
        result = mod.local_path_for_remote("myprefix/datasets/fineweb.bin")
        assert "myprefix" not in result.parts

    def test_datasets_subdir_preserved_in_path(self):
        result = mod.local_path_for_remote("datasets/fineweb10B_sp1024/fineweb_train_000000.bin")
        assert "fineweb10B_sp1024" in str(result)
        assert result.name == "fineweb_train_000000.bin"

    def test_unknown_prefix_falls_back_to_root(self):
        result = mod.local_path_for_remote("manifest.json")
        assert result.name == "manifest.json"


# ---------------------------------------------------------------------------
# artifact_paths_for_tokenizer
# ---------------------------------------------------------------------------

class TestArtifactPathsForTokenizer:
    def test_model_path_only(self):
        entry = {"name": "sp1024", "model_path": "tokenizers/model.model"}
        result = artifact_paths_for_tokenizer(entry)
        assert "tokenizers/model.model" in result

    def test_model_and_vocab_path(self):
        entry = {"model_path": "t/m.model", "vocab_path": "t/m.vocab"}
        result = artifact_paths_for_tokenizer(entry)
        assert len(result) == 2
        assert "t/m.model" in result
        assert "t/m.vocab" in result

    def test_all_three_artifact_keys(self):
        entry = {"model_path": "a", "vocab_path": "b", "path": "c"}
        result = artifact_paths_for_tokenizer(entry)
        assert set(result) == {"a", "b", "c"}

    def test_path_key_only(self):
        entry = {"path": "tokenizers/byte260.json"}
        result = artifact_paths_for_tokenizer(entry)
        assert result == ["tokenizers/byte260.json"]

    def test_missing_all_keys_raises(self):
        with pytest.raises(ValueError, match="missing downloadable artifacts"):
            artifact_paths_for_tokenizer({"name": "no-artifacts-here"})

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="missing downloadable artifacts"):
            artifact_paths_for_tokenizer({})

    def test_none_values_are_skipped(self):
        """Keys present but with None values should not count as artifacts."""
        entry = {"model_path": None, "vocab_path": None, "path": "real/path.bin"}
        result = artifact_paths_for_tokenizer(entry)
        assert result == ["real/path.bin"]


# ---------------------------------------------------------------------------
# build_parser (argument parsing)
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_default_train_shards(self):
        parser = mod.build_parser()
        args = parser.parse_args([])
        assert args.train_shards == 80

    def test_custom_train_shards(self):
        parser = mod.build_parser()
        args = parser.parse_args(["--train-shards", "10"])
        assert args.train_shards == 10

    def test_default_variant(self):
        parser = mod.build_parser()
        args = parser.parse_args([])
        assert args.variant == "sp1024"

    def test_custom_variant(self):
        parser = mod.build_parser()
        args = parser.parse_args(["--variant", "byte260"])
        assert args.variant == "byte260"

    def test_skip_manifest_flag(self):
        parser = mod.build_parser()
        args = parser.parse_args(["--skip-manifest"])
        assert args.skip_manifest is True

    def test_with_docs_flag(self):
        parser = mod.build_parser()
        args = parser.parse_args(["--with-docs"])
        assert args.with_docs is True

    def test_positional_train_shards_takes_precedence(self):
        """Legacy positional arg should override --train-shards default."""
        parser = mod.build_parser()
        args = parser.parse_args(["5"])
        assert args.train_shards_positional == 5

    def test_negative_train_shards_accepted_by_parser(self):
        """Parser accepts negatives; main() validates them."""
        parser = mod.build_parser()
        args = parser.parse_args(["--train-shards", "-1"])
        assert args.train_shards == -1
