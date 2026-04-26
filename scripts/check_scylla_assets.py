#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = [
    Path("tokenizer/candidate.vocab"),
    Path("tokenizer/candidate.meta.npz"),
    Path("datasets/fineweb10B_scylla/fineweb_train_000000.bin"),
    Path("datasets/fineweb10B_scylla/fineweb_val_000000.bin"),
]


def missing_assets(root: Path) -> list[Path]:
    return [rel for rel in REQUIRED_FILES if not (root / rel).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    missing = missing_assets(args.root)
    if missing:
        print("missing Scylla assets:")
        for rel in missing:
            print(f"  {args.root / rel}")
        return 1
    print("Scylla assets present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
