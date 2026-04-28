"""Run the PR1813 Scylla sub-1.05 reproduction on Modal 8xH100.

This intentionally does not use the older INT4/SP1024 `run_modal_sub105.py` lane.
It runs the exact Scylla config recorded in PR1813 and normalizes the public
Amarck Hugging Face dataset into the path layout expected by that script.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import modal


DATASET_REPO = "amarck/parameter-golf-scylla"
APP_NAME = "parameter-golf-scylla-pr1813"
VOLUME_NAME = "pg-scylla-pr1813-data"
TRAIN_SCRIPT_LOCAL = "frontier_sources/scylla_pr1813/train_gpt_modal.py"
TRAIN_SCRIPT_REMOTE = "/workspace/pg/train_gpt.py"
SAFE_SCYLLA_OVERRIDES = {
    "BIGRAM_DIM",
    "QK_GAIN_INIT",
    "ENABLE_LOOPING_AT",
    "LOOP_START",
    "LOOP_END",
    "NUM_LOOPS",
}


def scylla_reference_env(
    seed: int,
    data_root: str = "/data",
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = {
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor_pg",
        "TMPDIR": "/tmp",
        "RUN_ID": f"scylla_qk525_loop3to5_bg40_seed{seed}",
        "SEED": str(seed),
        "ITERATIONS": "9000",
        "VAL_LOSS_EVERY": "0",
        "DATA_PATH": f"{data_root}/datasets/fineweb10B_scylla",
        "TOKENIZER_PATH": f"{data_root}/tokenizer/candidate.vocab",
        "TOKENIZER_META_PATH": f"{data_root}/tokenizer/candidate.meta.npz",
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
    }
    if overrides:
        unsafe = sorted(set(overrides) - SAFE_SCYLLA_OVERRIDES)
        if unsafe:
            raise ValueError(f"unsafe Scylla override(s): {', '.join(unsafe)}")
        env.update({key: str(value) for key, value in overrides.items()})
        suffix = "_".join(f"{key.lower()}{env[key]}" for key in sorted(overrides))
        env["RUN_ID"] = f"{env['RUN_ID']}_{suffix}"
    return env


def scylla_override_env(
    *,
    bigram_dim: int | None = None,
    qk_gain: float | None = None,
    loop_at: float | None = None,
    loop_start: int | None = None,
    loop_end: int | None = None,
    num_loops: int | None = None,
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if bigram_dim is not None:
        overrides["BIGRAM_DIM"] = str(bigram_dim)
    if qk_gain is not None:
        overrides["QK_GAIN_INIT"] = str(qk_gain)
    if loop_at is not None:
        overrides["ENABLE_LOOPING_AT"] = str(loop_at)
    if loop_start is not None:
        overrides["LOOP_START"] = str(loop_start)
    if loop_end is not None:
        overrides["LOOP_END"] = str(loop_end)
    if num_loops is not None:
        overrides["NUM_LOOPS"] = str(num_loops)
    return overrides


def normalize_scylla_layout(root: Path) -> None:
    dataset_candidates = [
        root / "amarck_scylla" / "datasets" / "fineweb10B_scylla",
        root / "fineweb10B_scylla_raw",
        root / "fineweb_scylla",
        root / "datasets" / "fineweb10B_scylla",
    ]
    raw_dataset = next((path for path in dataset_candidates if path.is_dir()), None)
    if raw_dataset is None:
        searched = ", ".join(str(path) for path in dataset_candidates)
        raise FileNotFoundError(f"missing downloaded Scylla dataset directory; searched: {searched}")

    dataset_target = root / "datasets" / "fineweb10B_scylla"
    tokenizer_target = root / "tokenizer"
    dataset_target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_target.mkdir(parents=True, exist_ok=True)

    if dataset_target.is_symlink():
        if dataset_target.resolve() != raw_dataset.resolve():
            dataset_target.unlink()
    elif dataset_target.exists() and dataset_target.resolve() != raw_dataset.resolve():
        raise FileExistsError(f"dataset target exists and is not the selected Scylla source: {dataset_target}")
    if not dataset_target.exists():
        dataset_target.symlink_to(raw_dataset, target_is_directory=True)


def install_reference_tokenizer(reference_dir: Path, root: Path) -> None:
    sources = {
        "candidate.vocab": reference_dir / "candidate.vocab",
        "candidate.meta.npz": reference_dir / "candidate.pr1813_compat.meta.npz",
    }

    tokenizer_target = root / "tokenizer"
    tokenizer_target.mkdir(parents=True, exist_ok=True)
    for name, src in sources.items():
        if not src.is_file():
            raise FileNotFoundError(f"missing reference tokenizer file: {src}")
        dst = tokenizer_target / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


image = (
    modal.Image.from_registry("matotezitanka/proteus-pytorch:community")
    .pip_install("huggingface-hub", extra_options="--break-system-packages")
    .add_local_file(TRAIN_SCRIPT_LOCAL, TRAIN_SCRIPT_REMOTE, copy=True)
    .add_local_file("frontier_sources/scylla_pr1813/candidate.vocab", "/workspace/pg/reference_tokenizer/candidate.vocab", copy=True)
    .add_local_file("frontier_sources/scylla_pr1813/candidate.meta.npz", "/workspace/pg/reference_tokenizer/candidate.meta.npz", copy=True)
    .add_local_file("frontier_sources/scylla_pr1813/candidate.pr1813_compat.meta.npz", "/workspace/pg/reference_tokenizer/candidate.pr1813_compat.meta.npz", copy=True)
    .add_local_file("scripts/check_scylla_assets.py", "/workspace/pg/scripts/check_scylla_assets.py", copy=True)
    .add_local_file("scripts/check_scylla_artifact.py", "/workspace/pg/scripts/check_scylla_artifact.py", copy=True)
    .add_local_file("scripts/preflight_scylla.py", "/workspace/pg/scripts/preflight_scylla.py", copy=True)
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, timeout=7200, cpu=8)
def download_scylla_data() -> str:
    import subprocess

    root = Path("/data")
    marker = root / "amarck_scylla" / "datasets" / "fineweb10B_scylla" / "fineweb_train_000193.bin"
    hf_meta = root / "amarck_scylla" / "tokenizer" / "candidate.meta.npz"
    meta = root / "tokenizer" / "candidate.meta.npz"
    if marker.is_file() and hf_meta.is_file() and meta.is_file():
        normalize_scylla_layout(root)
        install_reference_tokenizer(Path("/workspace/pg/reference_tokenizer"), root)
        vol.commit()
        return "Scylla data already present; PR1813 tokenizer metadata installed"

    subprocess.run(
        [
            "hf",
            "download",
            DATASET_REPO,
            "--repo-type",
            "dataset",
            "--include",
            "datasets/fineweb10B_scylla/*",
            "--include",
            "tokenizer/*",
            "--local-dir",
            str(root / "amarck_scylla"),
        ],
        check=True,
    )
    normalize_scylla_layout(root)
    install_reference_tokenizer(Path("/workspace/pg/reference_tokenizer"), root)
    vol.commit()
    return "Scylla data downloaded and normalized"


@app.function(image=image, volumes={"/data": vol}, timeout=300, cpu=4)
def smoke_environment() -> dict[str, str]:
    import importlib.util
    import subprocess
    import sys

    normalize_scylla_layout(Path("/data"))
    install_reference_tokenizer(Path("/workspace/pg/reference_tokenizer"), Path("/data"))
    subprocess.run([sys.executable, "/workspace/pg/scripts/check_scylla_assets.py", "--root", "/data"], check=True)
    import numpy as np

    flash_spec = importlib.util.find_spec("flash_attn_interface")
    val_path = Path("/data/datasets/fineweb10B_scylla/fineweb_val_000000.bin")
    with np.load("/data/tokenizer/candidate.meta.npz", allow_pickle=False) as meta:
        base_bytes = np.asarray(meta["base_bytes"], dtype=np.int64)
        has_leading_space = np.asarray(meta["has_leading_space"], dtype=np.bool_)
        is_boundary_token = np.asarray(meta["is_boundary_token"], dtype=np.bool_)
    header_bytes = 256 * np.dtype("<i4").itemsize
    n_tokens = (val_path.stat().st_size - header_bytes) // np.dtype("<u2").itemsize
    val_tokens = np.memmap(val_path, mode="r", dtype="<u2", offset=header_bytes, shape=(n_tokens,))
    prev_ids = np.asarray(val_tokens[:-1], dtype=np.int64)
    tgt_ids = np.asarray(val_tokens[1:], dtype=np.int64)
    token_count = int(tgt_ids.size)
    token_bytes = base_bytes[tgt_ids].copy()
    token_bytes += (has_leading_space[tgt_ids] & ~is_boundary_token[prev_ids]).astype(np.int64)
    byte_count = int(token_bytes.sum())
    return {
        "python": sys.version.split()[0],
        "flash_attn_interface": "present" if flash_spec else "missing",
        "val_tokens": str(token_count),
        "val_bytes": str(byte_count),
        "tokens_per_byte": f"{token_count / byte_count:.9f}",
    }


@app.function(image=image, volumes={"/data": vol}, gpu="H100:8", timeout=3600, cpu=32, memory=240_000)
def train(
    seed: int = 1337,
    bigram_dim: int | None = None,
    qk_gain: float | None = None,
    loop_at: float | None = None,
    loop_start: int | None = None,
    loop_end: int | None = None,
    num_loops: int | None = None,
) -> dict[str, object]:
    import re
    import subprocess
    import sys

    workdir = Path("/workspace/pg")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "scripts").mkdir(parents=True, exist_ok=True)
    normalize_scylla_layout(Path("/data"))
    install_reference_tokenizer(Path("/workspace/pg/reference_tokenizer"), Path("/data"))

    overrides = scylla_override_env(
        bigram_dim=bigram_dim,
        qk_gain=qk_gain,
        loop_at=loop_at,
        loop_start=loop_start,
        loop_end=loop_end,
        num_loops=num_loops,
    )
    env = {**os.environ, **scylla_reference_env(seed, "/data", overrides)}
    subprocess.run([sys.executable, str(workdir / "scripts" / "check_scylla_assets.py"), "--root", "/data"], env=env, check=True)
    subprocess.run([sys.executable, str(workdir / "scripts" / "preflight_scylla.py")], env=env, check=True)

    log_path = workdir / f"scylla_seed{seed}.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["torchrun", "--standalone", "--nproc_per_node=8", TRAIN_SCRIPT_REMOTE],
            cwd=workdir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = proc.wait()
    if returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"training failed with exit code {returncode}\n{tail}")

    model = workdir / "final_model.int6.ptz"
    size_proc = subprocess.run(
        [
            sys.executable,
            str(workdir / "scripts" / "check_scylla_artifact.py"),
            "--model",
            str(model),
            "--code",
            TRAIN_SCRIPT_REMOTE,
            "--min-margin",
            "100000",
        ],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    sliding = re.findall(r"final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    roundtrip = re.findall(r"final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    return {
        "seed": seed,
        "artifact_bytes": model.stat().st_size if model.exists() else None,
        "sliding_bpb": float(sliding[-1]) if sliding else None,
        "roundtrip_bpb": float(roundtrip[-1]) if roundtrip else None,
        "size_check": size_proc.stdout,
        "log_tail": "\n".join(text.splitlines()[-40:]),
    }


@app.function(image=image, timeout=60)
def sweep() -> list[dict[str, object]]:
    return list(train.map([1337, 42, 2025]))


@app.local_entrypoint()
def main(download: bool = False, smoke: bool = False, seed: int = 1337, sweep3: bool = False) -> None:
    if download:
        print(download_scylla_data.remote())
    if smoke:
        print(smoke_environment.remote())
    if sweep3:
        print(sweep.remote())
    elif not download and not smoke:
        print(train.remote(seed))
