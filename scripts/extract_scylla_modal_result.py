#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


CAP_BYTES = 16_000_000


@dataclass(frozen=True)
class ScyllaModalResult:
    seed: int
    step: int
    train_time_ms: int
    model_bytes: int
    total_bytes: int
    margin_bytes: int
    roundtrip_bpb: float
    sliding_bpb: float


def _last_match(pattern: str, text: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        raise ValueError(f"missing {label} in Scylla Modal log")
    return matches[-1]


def parse_result(text: str, *, cap: int = CAP_BYTES) -> ScyllaModalResult:
    seed = int(_last_match(r"\bseed:(\d+)\b", text, "seed").group(1))
    step_match = _last_match(
        r"\bstep:(\d+)/\d+\s+val_loss:[0-9.]+\s+val_bpb:[0-9.]+\s+train_time:(\d+)ms\b",
        text,
        "final train step",
    )
    model_bytes = int(
        _last_match(r"Serialized model int6\+lzma:\s+(\d+)\s+bytes", text, "model bytes").group(1)
    )
    total_bytes = int(
        _last_match(r"Total submission size int6\+lzma:\s+(\d+)\s+bytes", text, "total bytes").group(1)
    )
    roundtrip_bpb = float(
        _last_match(
            r"final_int6_roundtrip_exact\s+val_loss:[0-9.]+\s+val_bpb:([0-9.]+)",
            text,
            "roundtrip BPB",
        ).group(1)
    )
    sliding_bpb = float(
        _last_match(
            r"final_int6_sliding_window_exact\s+val_loss:[0-9.]+\s+val_bpb:([0-9.]+)",
            text,
            "sliding BPB",
        ).group(1)
    )
    return ScyllaModalResult(
        seed=seed,
        step=int(step_match.group(1)),
        train_time_ms=int(step_match.group(2)),
        model_bytes=model_bytes,
        total_bytes=total_bytes,
        margin_bytes=cap - total_bytes,
        roundtrip_bpb=roundtrip_bpb,
        sliding_bpb=sliding_bpb,
    )


def result_to_json(result: ScyllaModalResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", type=Path, help="Modal log path; reads stdin when omitted")
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace") if args.log else sys.stdin.read()
    print(result_to_json(parse_result(text)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
