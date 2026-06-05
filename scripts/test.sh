#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:2
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/test_%j.log
#SBATCH --time=0-00:59:00
#SBATCH --job-name=clara_stage1_cr4

set -ex

# ─────────────────────────────────────────
# Environment
# ─────────────────────────────────────────
source /vol/cuda/12.9.0/setup.sh

export CUDA_HOME=/vol/cuda/12.9.0
export UV_CACHE_DIR=/vol/biomedic3/vd425/.cache/uv
export PATH=$CUDA_HOME/bin:$PATH
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate
export PYTHONPATH=/vol/biomedic3/vd425/clara-main/ml-clara:$PYTHONPATH

nvcc --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CR=4
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
BASE_DIR=/vol/biomedic3/vd425/clara-main
MODEL_PATH=$BASE_DIR/models/Mistral-7B-Instruct-v0.2
DATA_PATH=$BASE_DIR/ml-clara/example/pretrain_data.jsonl

SAVE_PATH=$BASE_DIR/checkpoints/stage1/cr${CR}
mkdir -p $SAVE_PATH $BASE_DIR/logs

echo "Using ${NUM_GPUS} GPUs"

# ─────────────────────────────────────────
# Training command
# ─────────────────────────────────────────
training_commands="openrlhf.cli.train_sft \
    --pretrain $MODEL_PATH \
    --stage stage1 \
    --compress_rate $CR \
    --doc_max_length 256 \
    --dataset $DATA_PATH \
    --max_len 2048 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --max_samples 500 \
    --save_path $SAVE_PATH \
    --ckpt_path $SAVE_PATH/checkpoints \
    --save_steps -2 \
    --save_hf_ckpt \
    --logging_steps 1 \
    --eval_steps 20 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 1e-4 \
    --lr_scheduler cosine_with_min_lr \
    --gradient_checkpointing \
    --generation_top_k 1 \
    --qa_loss \
    --mse_loss"

# ─────────────────────────────────────────
# Launch
# ─────────────────────────────────────────
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --rdzv_id 101 \
    --rdzv_backend c10d \
    --rdzv_endpoint 127.0.0.1:29501 \
    --master_addr 127.0.0.1 \
    --master_port 29501 \
    --node_rank 0"

nvidia-smi
echo "Starting Stage 1 training — compression rate ${CR}x on ${NUM_GPUS} GPUs"
torchrun $DISTRIBUTED_ARGS -m $training_commands
echo "Done!"