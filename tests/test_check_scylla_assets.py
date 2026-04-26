from scripts.check_scylla_assets import REQUIRED_FILES, missing_assets


def test_missing_assets_reports_all_required_files(tmp_path):
    missing = missing_assets(tmp_path)
    assert [p.name for p in missing] == [p.name for p in REQUIRED_FILES]


def test_missing_assets_accepts_complete_tree(tmp_path):
    for rel in REQUIRED_FILES:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    assert missing_assets(tmp_path) == []
