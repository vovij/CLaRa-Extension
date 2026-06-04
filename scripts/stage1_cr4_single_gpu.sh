#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/stage1_cr4_debug_%j.log
#SBATCH --time=0-00:59:00
#SBATCH --job-name=debug_clara_stage1_cr4

set -ex

# ─────────────────────────────────────────
# Environment
# ─────────────────────────────────────────
source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate

export PYTHONPATH=/vol/biomedic3/vd425/clara-main/ml-clara:$PYTHONPATH

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
CR=4
BASE_DIR=/vol/biomedic3/vd425/clara-main
MODEL_PATH=$BASE_DIR/models/Mistral-7B-Instruct-v0.2
DATA_PATH=$BASE_DIR/ml-clara/example/pretrain_data.jsonl  # example data

SAVE_PATH=$BASE_DIR/checkpoints/stage1/cr${CR}
mkdir -p $SAVE_PATH $BASE_DIR/logs

# ─────────────────────────────────────────
# Launch
# ─────────────────────────────────────────
nvidia-smi
echo "Starting Stage 1 training — compression rate ${CR}x"

python /vol/biomedic3/vd425/clara-main/ml-clara/openrlhf/cli/train_sft.py \
    --pretrain $MODEL_PATH \
    --stage stage1 \
    --compress_rate $CR \
    --doc_max_length 256 \
    --dataset $DATA_PATH \
    --max_len 2048 \
    --train_batch_size 1 \
    --micro_train_batch_size 1 \
    --save_path $SAVE_PATH \
    --ckpt_path $SAVE_PATH/checkpoints \
    --save_steps 2000 \
    --save_hf_ckpt \
    --logging_steps 1 \
    --eval_steps 20 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --learning_rate 1e-4 \
    --gradient_checkpointing \
    --generation_top_k 1 \
    --qa_loss \
    --mse_loss"


echo "Done!"
