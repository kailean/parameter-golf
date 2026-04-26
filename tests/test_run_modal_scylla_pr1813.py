from pathlib import Path

from run_modal_scylla_pr1813 import normalize_scylla_layout, scylla_reference_env


def test_scylla_reference_env_matches_pr1813():
    env = scylla_reference_env(1337, "/data")
    assert env["VOCAB_SIZE"] == "998"
    assert env["QK_GAIN_INIT"] == "5.25"
    assert env["BIGRAM_DIM"] == "40"
    assert env["TTT_ENABLED"] == "0"
    assert env["DATA_PATH"] == "/data/datasets/fineweb10B_scylla"


def test_normalize_scylla_layout_creates_expected_symlinks(tmp_path):
    raw_dataset = tmp_path / "fineweb_scylla"
    raw_tokenizer = tmp_path / "tokenizers" / "scylla"
    raw_dataset.mkdir(parents=True)
    raw_tokenizer.mkdir(parents=True)
    (raw_dataset / "fineweb_train_000000.bin").write_bytes(b"train")
    (raw_tokenizer / "candidate.vocab").write_text("vocab", encoding="utf-8")
    (raw_tokenizer / "candidate.meta.npz").write_bytes(b"meta")

    normalize_scylla_layout(tmp_path)

    assert (tmp_path / "datasets" / "fineweb10B_scylla" / "fineweb_train_000000.bin").is_file()
    assert (tmp_path / "tokenizer" / "candidate.vocab").is_file()
    assert (tmp_path / "tokenizer" / "candidate.meta.npz").is_file()
