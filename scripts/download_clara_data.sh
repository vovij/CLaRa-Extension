#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/download_clara_%j.log
#SBATCH --time=0-06:00:00
#SBATCH --job-name=download_clara

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate
export HF_HOME=/vol/biomedic3/vd425/.cache/huggingface

python - << 'EOF'
from huggingface_hub import snapshot_download
import os

OUT = "/vol/biomedic3/vd425/clara-main/data"
os.makedirs(OUT, exist_ok=True)

print("Downloading full repository...", flush=True)
snapshot_download(
    repo_id="apple/CLaRa_multi_stage",
    repo_type="dataset",
    local_dir=OUT,
)
print("✓ All done!", flush=True)
EOF