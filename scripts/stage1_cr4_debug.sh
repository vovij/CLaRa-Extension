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
export NCCL_DEBUG=INFO

# DeepSpeed — skip all CUDA op compilation (no nvcc on this cluster)
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_SKIP_CUDA_CHECK=1
export DS_ACCELERATOR=cpu

# Redirect all caches away from /homes (quota exceeded there)
export HF_HOME=/vol/biomedic3/vd425/.cache/huggingface
export HF_DATASETS_CACHE=/vol/biomedic3/vd425/.cache/huggingface/datasets
export TORCH_EXTENSIONS_DIR=/vol/biomedic3/vd425/.cache/torch_extensions
export TMPDIR=/vol/biomedic3/vd425/.tmp
mkdir -p $HF_DATASETS_CACHE $TORCH_EXTENSIONS_DIR $TMPDIR

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
# Training command
# ─────────────────────────────────────────
training_commands="openrlhf/cli/train_sft.py \
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

# ─────────────────────────────────────────
# Launch
# ─────────────────────────────────────────
nvidia-smi
echo "Starting Stage 1 training — compression rate ${CR}x"
uv run $training_commands
echo "Done!"