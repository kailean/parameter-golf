"""Tests for data/download_hf_docs_and_tokenize.py utility functions.

These tests cover pure, CPU-only utility logic that requires no network access,
no GPU, and no SentencePiece model file. They validate data-pipeline correctness
that would otherwise only be caught by a full end-to-end run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.download_hf_docs_and_tokenize import (
    DATAFILE_MAGIC,
    DATAFILE_VERSION,
    PureByteTokenizer,
    batched_docs_jsonl,
    count_docs,
    docs_sidecar_path,
    iter_docs,
    load_specs,
    maybe_load_docs_sidecar_meta,
    parse_reuse_sp_models,
    relativize_manifest_paths,
    tokenizer_kind,
    write_datafile,
)


# ---------------------------------------------------------------------------
# PureByteTokenizer
# ---------------------------------------------------------------------------

class TestPureByteTokenizer:
    def test_vocab_size_is_260(self):
        """vocab_size must be exactly 4 special tokens + 256 byte values."""
        tok = PureByteTokenizer()
        assert tok.vocab_size == 260

    def test_special_token_ids(self):
        tok = PureByteTokenizer()
        assert tok.pad_id == 0
        assert tok.bos_id == 1
        assert tok.eos_id == 2
        assert tok.unk_id == 3

    def test_encode_ascii_text(self):
        tok = PureByteTokenizer()
        tokens = tok.encode("hi")
        assert isinstance(tokens, np.ndarray)
        assert tokens.dtype == np.uint16
        # 'h'=104, 'i'=105 → each + byte_offset (4) = 108, 109
        np.testing.assert_array_equal(tokens, np.array([108, 109], dtype=np.uint16))

    def test_encode_empty_string(self):
        tok = PureByteTokenizer()
        assert len(tok.encode("")) == 0

    def test_encode_length_equals_utf8_byte_count(self):
        tok = PureByteTokenizer()
        text = "hello world"
        assert len(tok.encode(text)) == len(text.encode("utf-8"))

    def test_encode_multibyte_unicode(self):
        """'é' is 2 UTF-8 bytes, so 'café' should produce 5 tokens."""
        tok = PureByteTokenizer()
        tokens = tok.encode("café")
        assert len(tokens) == len("café".encode("utf-8"))  # 5

    def test_all_tokens_within_vocab_range(self):
        tok = PureByteTokenizer()
        tokens = tok.encode("test with various chars: 123 !@#\x00\xff")
        assert (tokens >= tok.byte_offset).all()
        assert (tokens < tok.vocab_size).all()

    def test_encode_batch_matches_individual_encode(self):
        tok = PureByteTokenizer()
        texts = ["hello", "world", ""]
        batch = tok.encode_batch(texts)
        assert len(batch) == 3
        np.testing.assert_array_equal(batch[0], tok.encode("hello"))
        np.testing.assert_array_equal(batch[1], tok.encode("world"))
        assert len(batch[2]) == 0

    def test_save_json_writes_valid_payload(self, tmp_path):
        tok = PureByteTokenizer()
        path = tmp_path / "tok.json"
        tok.save_json(path)
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["tokenizer_type"] == "pure_byte"
        assert payload["vocab_size"] == 260
        assert "config" in payload

    def test_save_json_creates_parent_dirs(self, tmp_path):
        tok = PureByteTokenizer()
        path = tmp_path / "sub" / "dir" / "tok.json"
        tok.save_json(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# write_datafile
# ---------------------------------------------------------------------------

class TestWriteDatafile:
    def _read_shard(self, path: Path):
        header = np.fromfile(path, dtype="<i4", count=256)
        num_tokens = int(header[2])
        tokens = np.fromfile(path, dtype="<u2", offset=256 * 4, count=num_tokens)
        return header, tokens

    def test_roundtrip_small_array(self, tmp_path):
        tokens = np.array([1, 2, 3, 100, 65535], dtype=np.uint16)
        path = tmp_path / "shard.bin"
        write_datafile(path, tokens)
        header, read_tokens = self._read_shard(path)
        assert int(header[0]) == DATAFILE_MAGIC
        assert int(header[1]) == DATAFILE_VERSION
        assert int(header[2]) == len(tokens)
        np.testing.assert_array_equal(read_tokens, tokens)

    def test_header_has_correct_magic_and_version(self, tmp_path):
        path = tmp_path / "shard.bin"
        write_datafile(path, np.array([0], dtype=np.uint16))
        header, _ = self._read_shard(path)
        assert int(header[0]) == DATAFILE_MAGIC
        assert int(header[1]) == DATAFILE_VERSION

    def test_rejects_token_count_at_2_to_31(self, tmp_path):
        class FakeToks:
            def __len__(self):
                return 2**31
        with pytest.raises(ValueError, match="token count too large"):
            write_datafile(tmp_path / "x.bin", FakeToks())

    def test_int32_input_is_accepted_and_converted(self, tmp_path):
        path = tmp_path / "shard.bin"
        tokens = np.array([0, 1, 1023], dtype=np.int32)
        write_datafile(path, tokens)
        _, read_tokens = self._read_shard(path)
        np.testing.assert_array_equal(read_tokens, np.array([0, 1, 1023], dtype=np.uint16))

    def test_rejects_negative_token_ids(self, tmp_path):
        path = tmp_path / "shard.bin"
        tokens = np.array([-1, 0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="token dictionary too large"):
            write_datafile(path, tokens)

    def test_empty_token_array(self, tmp_path):
        """Empty shards should be written without error."""
        path = tmp_path / "empty.bin"
        write_datafile(path, np.array([], dtype=np.uint16))
        header, read_tokens = self._read_shard(path)
        assert int(header[2]) == 0
        assert len(read_tokens) == 0


# ---------------------------------------------------------------------------
# relativize_manifest_paths
# ---------------------------------------------------------------------------

class TestRelativizeManifestPaths:
    def test_absolute_path_is_relativized(self, tmp_path):
        result = relativize_manifest_paths(str(tmp_path / "foo" / "bar.json"), tmp_path)
        assert result == "foo/bar.json"

    def test_relative_string_is_unchanged(self, tmp_path):
        result = relativize_manifest_paths("just_a_string", tmp_path)
        assert result == "just_a_string"

    def test_path_outside_root_is_unchanged(self, tmp_path):
        other = "/some/completely/other/path.json"
        assert relativize_manifest_paths(other, tmp_path) == other

    def test_dict_values_are_recursed(self, tmp_path):
        d = {"key": str(tmp_path / "a.bin"), "other": "hello"}
        result = relativize_manifest_paths(d, tmp_path)
        assert result["key"] == "a.bin"
        assert result["other"] == "hello"

    def test_list_elements_are_recursed(self, tmp_path):
        lst = [str(tmp_path / "a.bin"), "hello", str(tmp_path / "b.bin")]
        result = relativize_manifest_paths(lst, tmp_path)
        assert result == ["a.bin", "hello", "b.bin"]

    def test_nested_structure(self, tmp_path):
        data = {"datasets": [{"path": str(tmp_path / "ds" / "shard.bin")}]}
        result = relativize_manifest_paths(data, tmp_path)
        assert result["datasets"][0]["path"] == "ds/shard.bin"

    def test_non_string_scalars_pass_through(self, tmp_path):
        assert relativize_manifest_paths(42, tmp_path) == 42
        assert relativize_manifest_paths(None, tmp_path) is None
        assert relativize_manifest_paths(3.14, tmp_path) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# parse_reuse_sp_models
# ---------------------------------------------------------------------------

class TestParseReuseSpModels:
    def test_empty_input(self):
        assert parse_reuse_sp_models([]) == {}

    def test_single_entry(self, tmp_path):
        model_file = tmp_path / "m.model"
        model_file.touch()
        result = parse_reuse_sp_models([f"1024={model_file}"])
        assert set(result.keys()) == {1024}
        assert result[1024] == model_file

    def test_multiple_entries(self, tmp_path):
        f1 = tmp_path / "m1.model"
        f2 = tmp_path / "m2.model"
        f1.touch()
        f2.touch()
        result = parse_reuse_sp_models([f"1024={f1}", f"4096={f2}"])
        assert set(result.keys()) == {1024, 4096}

    def test_duplicate_vocab_size_raises(self, tmp_path):
        f = tmp_path / "m.model"
        f.touch()
        with pytest.raises(ValueError, match="duplicate"):
            parse_reuse_sp_models([f"1024={f}", f"1024={f}"])

    def test_values_are_resolved_paths(self, tmp_path):
        """Returned paths should be absolute (resolved)."""
        f = tmp_path / "m.model"
        f.touch()
        result = parse_reuse_sp_models([f"512={f}"])
        assert result[512].is_absolute()


# ---------------------------------------------------------------------------
# load_specs
# ---------------------------------------------------------------------------

class TestLoadSpecs:
    def test_list_format(self, tmp_path):
        specs = [{"kind": "byte", "name": "t1"}]
        config = tmp_path / "specs.json"
        config.write_text(json.dumps(specs))
        result = load_specs(config)
        assert len(result) == 1
        assert result[0]["name"] == "t1"

    def test_dict_with_tokenizer_specs_key(self, tmp_path):
        payload = {"tokenizer_specs": [{"kind": "byte", "name": "t1"}]}
        config = tmp_path / "specs.json"
        config.write_text(json.dumps(payload))
        result = load_specs(config)
        assert len(result) == 1

    def test_dict_with_tokenizers_key(self, tmp_path):
        payload = {"tokenizers": [{"kind": "byte", "name": "t1"}]}
        config = tmp_path / "specs.json"
        config.write_text(json.dumps(payload))
        result = load_specs(config)
        assert len(result) == 1

    def test_empty_list_raises(self, tmp_path):
        config = tmp_path / "specs.json"
        config.write_text("[]")
        with pytest.raises(ValueError, match="non-empty"):
            load_specs(config)

    def test_non_dict_entries_raise(self, tmp_path):
        config = tmp_path / "specs.json"
        config.write_text('["not_a_dict"]')
        with pytest.raises(ValueError):
            load_specs(config)

    def test_multiple_specs(self, tmp_path):
        specs = [{"kind": "byte"}, {"kind": "sentencepiece_bpe", "vocab_size": 1024}]
        config = tmp_path / "specs.json"
        config.write_text(json.dumps(specs))
        result = load_specs(config)
        assert len(result) == 2

    def test_returns_independent_copies(self, tmp_path):
        """Mutating returned dicts should not affect re-reads."""
        specs = [{"kind": "byte", "name": "t1"}]
        config = tmp_path / "specs.json"
        config.write_text(json.dumps(specs))
        result = load_specs(config)
        result[0]["name"] = "modified"
        result2 = load_specs(config)
        assert result2[0]["name"] == "t1"


# ---------------------------------------------------------------------------
# tokenizer_kind
# ---------------------------------------------------------------------------

class TestTokenizerKind:
    def test_byte_kind_values(self):
        assert tokenizer_kind({"kind": "byte"}) == "byte"
        assert tokenizer_kind({"kind": "pure_byte"}) == "byte"

    def test_sentencepiece_kind_values(self):
        assert tokenizer_kind({"kind": "sentencepiece_bpe"}) == "sentencepiece_bpe"
        assert tokenizer_kind({"kind": "sentencepiece"}) == "sentencepiece_bpe"

    def test_builder_name_byte(self):
        assert tokenizer_kind({"builder": "mod:build_pure_byte_tokenizer"}) == "byte"

    def test_builder_name_sentencepiece(self):
        assert tokenizer_kind({"builder": "mod:build_sentencepiece_tokenizer"}) == "sentencepiece_bpe"

    def test_dataset_suffix_byte260(self):
        assert tokenizer_kind({"dataset_suffix": "byte260"}) == "byte"

    def test_vocab_size_implies_sentencepiece(self):
        assert tokenizer_kind({"vocab_size": 1024}) == "sentencepiece_bpe"

    def test_unknown_spec_raises(self):
        with pytest.raises(ValueError):
            tokenizer_kind({"name": "mystery_tokenizer"})

    def test_unknown_spec_error_includes_name(self):
        with pytest.raises(ValueError, match="mystery_tokenizer"):
            tokenizer_kind({"name": "mystery_tokenizer"})


# ---------------------------------------------------------------------------
# batched_docs_jsonl
# ---------------------------------------------------------------------------

class TestBatchedDocsJsonl:
    def _write_jsonl(self, path: Path, texts: list[str]) -> None:
        path.write_text(
            "\n".join(json.dumps({"text": t}) for t in texts) + "\n",
            encoding="utf-8",
        )

    def test_exact_batches(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        self._write_jsonl(path, ["a", "b", "c", "d"])
        batches = list(batched_docs_jsonl(path, batch_size=2))
        assert batches == [["a", "b"], ["c", "d"]]

    def test_partial_last_batch_is_included(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        self._write_jsonl(path, ["a", "b", "c"])
        batches = list(batched_docs_jsonl(path, batch_size=2))
        assert batches == [["a", "b"], ["c"]]

    def test_batch_size_larger_than_total(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        self._write_jsonl(path, ["x", "y"])
        batches = list(batched_docs_jsonl(path, batch_size=100))
        assert batches == [["x", "y"]]

    def test_single_document(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        self._write_jsonl(path, ["only_doc"])
        batches = list(batched_docs_jsonl(path, batch_size=1))
        assert batches == [["only_doc"]]

    def test_all_docs_are_yielded(self, tmp_path):
        texts = [str(i) for i in range(7)]
        path = tmp_path / "docs.jsonl"
        self._write_jsonl(path, texts)
        all_docs = [doc for batch in batched_docs_jsonl(path, batch_size=3) for doc in batch]
        assert all_docs == texts


# ---------------------------------------------------------------------------
# iter_docs / count_docs
# ---------------------------------------------------------------------------

class TestIterAndCountDocs:
    def _write_jsonl(self, path: Path, texts: list[str]) -> None:
        path.write_text(
            "\n".join(json.dumps({"text": t}) for t in texts) + "\n",
            encoding="utf-8",
        )

    def test_iter_docs_yields_text_fields(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        self._write_jsonl(path, ["hello", "world"])
        assert list(iter_docs(path)) == ["hello", "world"]

    def test_count_docs(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        self._write_jsonl(path, ["a", "b", "c"])
        assert count_docs(path) == 3

    def test_count_empty_file(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        path.write_text("")
        assert count_docs(path) == 0


# ---------------------------------------------------------------------------
# docs_sidecar_path / maybe_load_docs_sidecar_meta
# ---------------------------------------------------------------------------

class TestDocsSidecar:
    def test_sidecar_path_derivation(self, tmp_path):
        jsonl = tmp_path / "docs_selected.jsonl"
        expected = tmp_path / "docs_selected.source_manifest.json"
        assert docs_sidecar_path(jsonl) == expected

    def test_maybe_load_returns_none_when_file_absent(self, tmp_path):
        jsonl = tmp_path / "docs_selected.jsonl"
        assert maybe_load_docs_sidecar_meta(jsonl) is None

    def test_maybe_load_returns_dict_when_present(self, tmp_path):
        jsonl = tmp_path / "docs_selected.jsonl"
        sidecar = docs_sidecar_path(jsonl)
        sidecar.write_text(json.dumps({"num_docs": 100, "docs_val": 10}))
        result = maybe_load_docs_sidecar_meta(jsonl)
        assert result == {"num_docs": 100, "docs_val": 10}

    def test_maybe_load_raises_on_non_dict_json(self, tmp_path):
        jsonl = tmp_path / "docs_selected.jsonl"
        sidecar = docs_sidecar_path(jsonl)
        sidecar.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="JSON object"):
            maybe_load_docs_sidecar_meta(jsonl)
