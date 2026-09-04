#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/eval_trial_%j.log
#SBATCH --time=0-01:59:00
#SBATCH --job-name=clara_trial

source /vol/biomedic3/vd425/clara-main/miniconda3/etc/profile.d/conda.sh
conda activate clara-eval

export CUDA_HOME=/usr/local/cuda
pip install flash-attn --no-build-isolation -q

cd /vol/biomedic3/vd425/clara-main/ml-clara/evaluation

unset PYTHONPATH
SAVE_PATH=/vol/biomedic3/vd425/clara-main/checkpoints/CLaRa-7B-E2E/compression-128
SAVE_MODEL_NAME=${SAVE_PATH##*/}
export PYTHONPATH="$SAVE_PATH:/vol/biomedic3/vd425/clara-main/ml-clara:$PYTHONPATH"

for i in musique; do
    accelerate launch \
        --num_processes=1 \
        --mixed_precision=bf16 \
        evaluate.py \
        --model_path $SAVE_MODEL_NAME \
        --stage stage2 \
        --dataset $i \
        --gold_retrieval
done
echo "✅ All steps completed successfully"