#!/bin/bash
# Debug script: verify CUDA_VISIBLE_DEVICES is set correctly.
# Usage: ./test_gpu_device.sh [gpu_id]
#   gpu_id: GPU index (0-based, default 6). Example: ./test_gpu_device.sh 6
#
# Note: When CUDA_VISIBLE_DEVICES=6, PyTorch/CUDA report device 0 internally
#       because the process only "sees" one GPU (physical GPU 6). So
#       torch.cuda.current_device()=0 is correct; nvidia-smi shows physical GPU 6.

set -e
GPU_ID="${1:-6}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "=== Shell: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
echo ""

# Run from project root (parent of run_sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python - << 'PYEOF'
import os
import time

print("=== Python process ===")
cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES")
print(f"  os.environ['CUDA_VISIBLE_DEVICES'] = {repr(cuda_dev)}")
print()

try:
    import torch
    if torch.cuda.is_available():
        # Allocate small tensor so this process shows in nvidia-smi
        torch.zeros(1, device="cuda:0")
        print(f"  GPU memory allocated. Process should appear on physical GPU {cuda_dev} in nvidia-smi.")
    else:
        print("  CUDA not available; process will not appear in nvidia-smi.")
except ImportError:
    print("  PyTorch not installed; process will not appear in nvidia-smi.")

print()
print(">>> Process keeps running. Run 'nvidia-smi' in another terminal to see which physical GPU is used.")
print(">>> Press Ctrl+C to exit.")
print()

while True:
    time.sleep(60)
PYEOF
