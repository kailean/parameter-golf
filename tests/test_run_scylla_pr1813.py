import os
import subprocess
from pathlib import Path


def test_launcher_dry_run_renders_reference_env():
    script = Path("scripts/run_scylla_pr1813.sh")
    env = dict(os.environ, DRY_RUN="1", SEED="1337", DATA_ROOT="/workspace/pg/data")
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    assert result.returncode == 0
    assert "VOCAB_SIZE=998" in result.stdout
    assert "QK_GAIN_INIT=5.25" in result.stdout
    assert "BIGRAM_DIM=40" in result.stdout
    assert "torchrun --standalone --nproc_per_node=8 frontier_sources/scylla_pr1813/train_gpt.py" in result.stdout
