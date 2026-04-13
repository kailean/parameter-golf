#!/bin/bash
# RunPod 8×H100 Deployment Script for OmniClaw Parameter Golf
# Run this on a fresh RunPod instance with 8×H100

set -euo pipefail

echo "=== OmniClaw RunPod Setup ==="
echo "Starting at $(date)"

# 1. System setup
apt-get update && apt-get install -y git-lfs python3-venv tmux htop nvtop
git lfs install

# 2. Clone parameter-golf repo
cd /workspace
if [ ! -d parameter-golf ]; then
    git clone https://github.com/openai/parameter-golf.git
    cd parameter-golf
    git remote add kailean https://github.com/kailean/parameter-golf.git || true
    git fetch kailean
    git checkout kailean/submission-v1
else
    cd parameter-golf
    git pull --rebase
fi

# 3. Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install sentencepiece numpy huggingface_hub triton flash-attn

# 4. Download SP8192 data
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80

# 5. Verify GPU setup
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# 6. Run training with our config
echo "=== Ready to train ==="
echo "Run: bash run_omniclaw_v4.sh"
echo "Or for 3-seed verification: bash run_omniclaw_v4.sh --seeds 42 1337 2024"

echo "Setup complete at $(date)"