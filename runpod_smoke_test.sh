#!/bin/bash
# RunPod GPU Smoke Test for Parameter Golf
# Deploys 1x T4 pod and runs train_gpt_kl.py
# Prerequisites: RunPod account with credits, API key

set -euo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
if [ -z "$RUNPOD_API_KEY" ]; then
  echo "Set RUNPOD_API_KEY environment variable"
  echo "Get it from: https://runpod.io/console/user/settings"
  exit 1
fi

echo "=== Parameter Golf RunPod Smoke Test ==="

# 1. Create T4 pod
echo "[1/4] Creating RunPod pod with T4 GPU..."
POD_RESPONSE=$(curl -s -X POST "https://api.runpod.io/v2/pods" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "pg-smoke-t4",
    "imageName": "runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
    "gpuTypeId": "NVIDIA RTX A4000",
    "cloudType": "COMMUNITY",
    "containerDiskInGb": 50,
    "volumeInGb": 50,
    "minVcpuCount": 4,
    "minMemoryInGb": 15,
    "startSsh": true,
    "ports": "8888/http",
    "env": [
      {"key": "MAX_WALLCLOCK_SECONDS", "value": "300"},
      {"key": "ITERATIONS", "value": "100"}
    ]
  }')

POD_ID=$(echo "$POD_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")
echo "Pod ID: $POD_ID"

if [ -z "$POD_ID" ]; then
  echo "Failed to create pod. Response: $POD_RESPONSE"
  exit 1
fi

# 2. Wait for pod to be ready
echo "[2/4] Waiting for pod to start..."
for i in $(seq 1 60); do
  STATUS=$(curl -s "https://api.runpod.io/v2/pods/$POD_ID" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" | \
    python3 -c "import sys,json; print(json.load(sys.stdin).get('runtime',{}).get('status',''))" 2>/dev/null)
  if [ "$STATUS" = "RUNNING" ]; then
    echo "Pod is running!"
    break
  fi
  echo "  Status: $STATUS (waiting...)"
  sleep 10
done

# 3. Get SSH connection details
echo "[3/4] Getting connection details..."
POD_INFO=$(curl -s "https://api.runpod.io/v2/pods/$POD_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY")
echo "$POD_INFO" | python3 -c "
import sys,json
data=json.load(sys.stdin)
runtime=data.get('runtime',{})
print(f'SSH: ssh root@{runtime.get(\"ipAddress\",\"\")}:{runtime.get(\"ports\",[{}])[0].get(\"publicPort\",\"\")}'  if runtime.get('ipAddress') else 'SSH not ready yet')
"

# 4. Run smoke test via cloud-init/command
echo "[4/4] To run training, SSH into the pod and execute:"
echo "  cd /workspace && git clone https://github.com/kailean/parameter-golf.git"
echo "  cd parameter-golf && pip install sentencepiece brotli zstandard"
echo "  python3 train_gpt_kl.py"
echo ""
echo "To terminate pod: curl -X DELETE 'https://api.runpod.io/v2/pods/$POD_ID' -H 'Authorization: Bearer $RUNPOD_API_KEY'"
