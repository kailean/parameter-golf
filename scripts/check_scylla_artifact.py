#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SizeReport:
    model_bytes: int
    code_bytes: int
    total_bytes: int
    margin_bytes: int
    ok: bool


def check_size(model_path: Path, code_paths: list[Path], cap: int, min_margin: int) -> SizeReport:
    model_bytes = model_path.stat().st_size
    code_bytes = sum(path.stat().st_size for path in code_paths)
    total = model_bytes + code_bytes
    margin = cap - total
    return SizeReport(model_bytes, code_bytes, total, margin, margin >= min_margin)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("final_model.int6.ptz"))
    parser.add_argument("--code", type=Path, action="append", required=True)
    parser.add_argument("--cap", type=int, default=16_000_000)
    parser.add_argument("--min-margin", type=int, default=100_000)
    args = parser.parse_args()
    report = check_size(args.model, args.code, args.cap, args.min_margin)
    print(f"model_bytes={report.model_bytes}")
    print(f"code_bytes={report.code_bytes}")
    print(f"total_bytes={report.total_bytes}")
    print(f"margin_bytes={report.margin_bytes}")
    if not report.ok:
        print("FAIL: artifact margin below required threshold")
        return 1
    print("PASS: artifact size within cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
