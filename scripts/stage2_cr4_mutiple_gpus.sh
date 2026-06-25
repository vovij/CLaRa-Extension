#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:4
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/stage2_e2e_cr4_%j.log
#SBATCH --time=4-11:59:00
#SBATCH --job-name=clara_stage2_e2e_cr4

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
export HF_HOME=/vol/biomedic3/vd425/.cache/huggingface
export HF_DATASETS_CACHE=/vol/biomedic3/vd425/.cache/huggingface/datasets
export TRITON_CACHE_DIR=/vol/biomedic3/vd425/.cache/triton
export TORCH_EXTENSIONS_DIR=/vol/biomedic3/vd425/.cache/torch_extensions
export TMPDIR=/vol/biomedic3/vd425/.tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TMPDIR

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate
export PYTHONPATH=/vol/biomedic3/vd425/clara-main/ml-clara:$PYTHONPATH

nvcc --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CR=4
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
BASE_DIR=/vol/biomedic3/vd425/clara-main

MODEL_PATH=$BASE_DIR/models/Mistral-7B-Instruct-v0.2
PRETRAIN_CHECKPOINT=$BASE_DIR/checkpoints/stage1_2/cr${CR}

# Training data: all 4 datasets with positives guaranteed (paper setup).
# MuSiQue is hard to converge alone, so the paper combines NQ + HotpotQA +
# 2Wiki + MuSiQue into one pool. The *_with_pos variant ensures each sample
# has at least one gold document in its 20-candidate pool.
DATA_PATH="$BASE_DIR/data/e2e_training_ours_*_with_pos/train-*.parquet"

SAVE_PATH=$BASE_DIR/checkpoints/stage2/cr${CR}
mkdir -p $SAVE_PATH $BASE_DIR/logs

echo "Using ${NUM_GPUS} GPUs"
echo "Pretrain checkpoint: ${PRETRAIN_CHECKPOINT}"
echo "Train data: ${DATA_PATH}"

# ─────────────────────────────────────────
# Training command
# ─────────────────────────────────────────
training_commands=(
    openrlhf.cli.train_sft
    --pretrain "$MODEL_PATH"
    --pretrain_checkpoint "$PRETRAIN_CHECKPOINT"
    --stage stage2
    --compress_rate "$CR"
    --doc_max_length 256
    --dataset "$DATA_PATH"
    --max_len 1024
    --train_batch_size 32
    --micro_train_batch_size 1
    --save_path "$SAVE_PATH"
    --ckpt_path "$SAVE_PATH/checkpoints"
    --save_steps -2
    --save_hf_ckpt
    --logging_steps 1
    --eval_steps 100
    --zero_stage 2
    --max_epochs 1
    --bf16
    --flash_attn
    --learning_rate 5e-6
    --lr_scheduler constant
    --gradient_checkpointing
    --generation_top_k 5
    --qa_loss
    --do_eval_gen
)

# ─────────────────────────────────────────
# Launch
# ─────────────────────────────────────────
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --rdzv_id 103 \
    --rdzv_backend c10d \
    --rdzv_endpoint 127.0.0.1:29503 \
    --master_addr 127.0.0.1 \
    --master_port 29503 \
    --node_rank 0"

nvidia-smi
echo "Starting Stage 2 (end-to-end) training — compression rate ${CR}x on ${NUM_GPUS} GPUs"
torchrun $DISTRIBUTED_ARGS -m "${training_commands[@]}"

# Copy model definition alongside checkpoint
cp $BASE_DIR/ml-clara/openrlhf/models/modeling_clara.py $SAVE_PATH

echo "Done!"