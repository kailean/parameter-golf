#!/bin/bash
# Azure GPU Smoke Test for Parameter Golf
# Deploys NC4as_T4_v3 (1x T4 16GB) and runs train_gpt_kl.py
# Prerequisites: Azure CLI logged in, NCASv3_T4 quota approved (4+ vCPUs)

set -euo pipefail

RG="Parameter-Golf"
LOCATION="westeurope"
VM_NAME="pg-smoke-t4"
VM_SIZE="Standard_NC4as_T4_v3"
ADMIN_USER="omniclaw"
IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts:latest"

echo "=== Parameter Golf Azure Smoke Test ==="
echo "VM: $VM_SIZE in $LOCATION"
echo ""

# 1. Create VM
echo "[1/5] Creating GPU VM..."
az vm create \
  --name "$VM_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --image "$IMAGE" \
  --size "$VM_SIZE" \
  --admin-username "$ADMIN_USER" \
  --generate-ssh-keys \
  --priority Spot \
  --eviction-policy Deallocate \
  --max-price 0.15 \
  2>&1 || { echo "VM creation failed. Check quota: az vm list-usage --location $LOCATION | grep -i T4"; exit 1; }

VM_IP=$(az vm show --name "$VM_NAME" --resource-group "$RG" -d --query publicIps -o tsv)
echo "VM IP: $VM_IP"

# 2. Install CUDA + PyTorch
echo "[2/5] Installing CUDA toolkit and PyTorch..."
ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$VM_IP" << 'REMOTE'
set -e

# Install CUDA
sudo apt-get update -y
sudo apt-get install -y wget linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -y
sudo apt-get install -y cuda-12-4 cuda-drivers
sudo apt-get install -y python3-pip python3-venv git

# Reboot for NVIDIA driver
echo "NVIDIA driver installed. Will set up Python after reboot."
REMOTE

echo "[3/5] Rebooting VM for NVIDIA driver..."
az vm restart --name "$VM_NAME" --resource-group "$RG"
sleep 30

# 3. Verify GPU + install PyTorch
echo "[4/5] Verifying GPU and installing PyTorch..."
ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$VM_IP" << 'REMOTE'
set -e
nvidia-smi

# Install PyTorch with CUDA 12.4
python3 -m venv ~/pg-venv
source ~/pg-venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install sentencepiece numpy brotli zstandard

# Verify PyTorch CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Clone parameter-golf
cd ~
git clone https://github.com/kailean/parameter-golf.git || true
cd parameter-golf
REMOTE

# 4. Run smoke test
echo "[5/5] Running smoke test..."
ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$VM_IP" << 'REMOTE'
source ~/pg-venv/bin/activate
cd ~/parameter-golf

# Download tokenizer and data
echo "Downloading data..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('kevclark/parameter-golf', local_dir='./data', allow_patterns=['tokenizers/*', 'datasets/fineweb10B_sp1024/fineweb_val_*'])
" 2>/dev/null || {
  echo "HF download failed, trying manual..."
  mkdir -p data/tokenizers data/datasets/fineweb10B_sp1024
}

# Run training with reduced settings for smoke test
MAX_WALLCLOCK_SECONDS=300 ITERATIONS=100 python3 train_gpt_kl.py \
  2>&1 | tee smoke_test.log

echo "=== Smoke test complete ==="
REMOTE

echo ""
echo "=== Done. To SSH into VM: ssh $ADMIN_USER@$VM_IP ==="
echo "=== To delete VM when done: az vm delete --name $VM_NAME --resource-group $RG --yes ==="
