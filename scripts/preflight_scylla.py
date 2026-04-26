#!/usr/bin/env python3
from __future__ import annotations

import os


REQUIRED = {
    "VOCAB_SIZE": "998",
    "XSA_LAST_N": "11",
    "USE_GPTQ": "1",
    "GPTQ_RESERVE_MS": "9000",
    "TTT_ENABLED": "0",
    "BIGRAM_VOCAB_SIZE": "2816",
    "BIGRAM_DIM": "40",
    "QK_GAIN_INIT": "5.25",
    "NUM_LOOPS": "2",
    "LOOP_START": "3",
    "LOOP_END": "5",
    "ENABLE_LOOPING_AT": "0.35",
    "VAL_LOSS_EVERY": "0",
}


FORBIDDEN_TRUE = [
    "USE_PPM",
    "PPM_ENABLED",
    "USE_CACHE",
    "NGRAM_CACHE",
    "SLOT_ENABLED",
    "ETLB_ENABLED",
]


def validate_env(env: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for key, expected in REQUIRED.items():
        actual = env.get(key)
        if actual != expected:
            errors.append(f"{key} must be {expected}, got {actual!r}")
    for key in FORBIDDEN_TRUE:
        if env.get(key, "0") not in {"", "0", "false", "False"}:
            errors.append(f"{key} must be disabled for Scylla reproduction")
    return errors


def main() -> int:
    errors = validate_env(dict(os.environ))
    if errors:
        print("Scylla preflight failed:")
        for error in errors:
            print(f"  {error}")
        return 1
    print("Scylla preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
