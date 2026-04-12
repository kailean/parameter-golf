#!/bin/bash
# Azure H100 Deployment Script for OmniClaw Parameter Golf
# Run this on an Azure ND H100 v2 VM after connecting via SSH
# Azure credits: $200 (30-day expiry)
# H100 VM pricing: ~$3.60/hr for 1x H100, ~$28.80/hr for 8x H100

set -euo pipefail

echo "=== OmniClaw Azure H100 Setup ==="
echo "Starting at $(date)"

# 1. System setup
sudo apt-get update && sudo apt-get install -y git-lfs python3-venv python3-pip tmux htop nvtop
git lfs install

# 2. Clone parameter-golf repo
cd /workspace
if [ ! -d parameter-golf ]; then
    git clone https://github.com/kailean/parameter-golf.git
    cd parameter-golf
else
    cd parameter-golf
    git pull --rebase
fi

# 3. Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install sentencepiece numpy huggingface_hub triton flash-attn --no-build-isolation
pip install zstandard

# 4. Download SP8192 data
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80

# 5. Verify GPU setup
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# 6. Run training with our config
echo "=== Ready to train ==="
echo "Run: bash run_omniclaw_v4.sh"
echo "Or for 3-seed verification: bash run_omniclaw_v4.sh --seeds 42 1337 2024"

echo "Setup complete at $(date)"