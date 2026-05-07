#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic3/vd425/logs/download_%j.log
#SBATCH --time=0-04:00:00
#SBATCH --job-name=hf_download

export HF_TOKEN=$(cat /vol/biomedic3/vd425/.hf_token)

source /vol/biomedic3/vd425/miniconda3/bin/activate clara

hf download mistralai/Mistral-7B-Instruct-v0.2 \
    --local-dir /vol/biomedic3/vd425/models/Mistral-7B-Instruct-v0.2

hf download apple/CLaRa-7B-E2E \
    --local-dir /vol/biomedic3/vd425/checkpoints/CLaRa-7B-E2E

echo "All downloads complete!"
