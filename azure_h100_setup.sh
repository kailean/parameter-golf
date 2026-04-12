#!/bin/bash
# Azure H100 VM Setup — Request quota and deploy 8×H100
# Run this script after quota is approved

set -euo pipefail

echo "=== OmniClaw Azure H100 Setup ==="

# ── Configuration ──
SUBSCRIPTION_ID="be09f296-6b13-49a2-9e3e-ac3da9a8dfb1"
RESOURCE_GROUP="omniclaw-rg"
LOCATION="eastus"
VM_NAME="omniclaw-h100"
VM_SIZE="Standard_ND96isr_H100_v5"

# ── Check quota ──
echo "Checking H100 quota in $LOCATION..."
CURRENT=$(az vm list-usage --location $LOCATION --query "[?contains(name.value, 'NDSH100')].currentValue" -o tsv 2>/dev/null || echo "0")
LIMIT=$(az vm list-usage --location $LOCATION --query "[?contains(name.value, 'NDSH100')].limit" -o tsv 2>/dev/null || echo "0")
echo "Current H100 vCPU quota: $CURRENT / $LIMIT"

if [ "$LIMIT" -lt 96 ]; then
    echo ""
    echo "❌ H100 quota insufficient ($LIMIT vCPUs, need 96)"
    echo ""
    echo "Request quota increase at:"
    echo "  https://aka.ms/ProdportalCRP/#blade/Microsoft_Azure_Capacity/UsageAndQuota.ReactView"
    echo ""
    echo "Select:"
    echo "  Region: East US"
    echo "  SKU: Standard_ND96isr_H100_v5"  
    echo "  New Limit: 96"
    echo "  Reason: 'ML training competition — need 8×H100 for 10-minute run'"
    echo ""
    echo "After approval, re-run this script."
    exit 1
fi

echo "✅ Quota available: $LIMIT vCPUs"

# ── Create VM ──
echo "Creating VM $VM_NAME ($VM_SIZE)..."
az vm create \
    --resource-group $RESOURCE_GROUP \
    --name $VM_NAME \
    --image Ubuntu2204 \
    --size $VM_SIZE \
    --os-disk-size 256 \
    --os-disk-premium-storage \
    --ssh-key-files ~/.ssh/id_rsa.pub \
    --no-wait

echo "VM creation initiated. Once running, connect with:"
echo "  ssh azureuser@$(az vm show -d -g $RESOURCE_GROUP -n $VM_NAME --query publicIps -o tsv)"

echo ""
echo "=== Setup Instructions (run on VM) ==="
echo "After SSH:"
echo "  1. git clone -b kailean/submission-v1 https://github.com/kailean/parameter-golf.git"
echo "  2. cd parameter-golf"
echo "  3. bash azure_setup.sh"
echo "  4. bash run_h100.sh"