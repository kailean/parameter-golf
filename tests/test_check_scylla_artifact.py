from pathlib import Path

from scripts.check_scylla_artifact import SizeReport, check_size


def write_bytes(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n)
    return path


def test_check_size_passes_under_decimal_cap(tmp_path):
    model = write_bytes(tmp_path / "final_model.int6.ptz", 15_000_000)
    code = write_bytes(tmp_path / "train_gpt.py", 100_000)
    report = check_size(model, [code], cap=16_000_000, min_margin=100_000)
    assert report.ok
    assert report.total_bytes == 15_100_000


def test_check_size_fails_when_margin_too_small(tmp_path):
    model = write_bytes(tmp_path / "final_model.int6.ptz", 15_850_000)
    code = write_bytes(tmp_path / "train_gpt.py", 105_000)
    report = check_size(model, [code], cap=16_000_000, min_margin=100_000)
    assert not report.ok
    assert report.margin_bytes == 45_000
