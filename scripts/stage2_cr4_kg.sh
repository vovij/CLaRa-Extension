#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:8
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/stage2_cr4_kg_%j.log
#SBATCH --time=4-11:59:00
#SBATCH --job-name=clara_stage2_cr4_kg

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

# Fail fast if the new module isn't where it needs to be, rather than 40 minutes
# into model loading on 4 GPUs.
python -c "from openrlhf.models.temporal_link_matrix import TemporalLinkMatrix, multi_hop_propagate" \
    || { echo "ERROR: openrlhf/models/temporal_link_matrix.py not found/importable. Aborting."; exit 1; }

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CR=4
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
BASE_DIR=/vol/biomedic3/vd425/clara-main

MODEL_PATH=$BASE_DIR/models/Mistral-7B-Instruct-v0.2
PRETRAIN_CHECKPOINT=$BASE_DIR/checkpoints/stage1_2/cr${CR}

# Same data as the baseline run -- identical pool, identical positives-guaranteed
# setup, so the ONLY difference between this run and stage2_e2e_cr4.sh is the
# KG flags and the widened candidate pool below. That's what makes the two
# runs a valid A/B comparison.
DATA_PATH="$BASE_DIR/data/e2e_training_ours_*_with_pos/train-*.parquet"

SAVE_PATH=$BASE_DIR/checkpoints/stage2_kg/cr${CR}
mkdir -p $SAVE_PATH $BASE_DIR/logs

echo "Using ${NUM_GPUS} GPUs"
echo "Pretrain checkpoint: ${PRETRAIN_CHECKPOINT}  (stage1_2 -- link_module has no prior"
echo "  weights here and will train from a fresh random init during this run, same as"
echo "  query_reasoner_adapter does on any first stage2 run)"
echo "Train data: ${DATA_PATH}"

# ─────────────────────────────────────────
# Training command
# ─────────────────────────────────────────
# Differences from the non-KG baseline (stage2_e2e_cr4.sh), and why:
#
#   --stage2_retrieval_top_n 100   Widened candidate pool. The KG module needs a
#                                  real graph to link across; at the baseline's
#                                  implicit top_n (== generation_top_k == 5) there
#                                  are only 5 candidates and almost nothing to hop
#                                  between. 100 matches the paper's worked example.
#   --generation_top_k 5           UNCHANGED from baseline, deliberately. This is
#                                  NOT a free knob -- the training data
#                                  (e2e_training_ours_*_with_pos) was built with a
#                                  FIXED 5 documents per example, and the collate
#                                  function's decoder prompt construction bakes in
#                                  exactly that many <MEM> token blocks per example
#                                  at data-loading time, independent of any KG
#                                  changes. Raising this without matching training
#                                  data causes a slot-count mismatch between the
#                                  prompt's <MEM> blocks and the number of document
#                                  embeddings the model tries to write into them.
#                                  The KG module widens the CANDIDATE POOL it scores
#                                  over (stage2_retrieval_top_n), not the final
#                                  selection count fed to the generator -- those are
#                                  separate knobs; only the former is safe to widen
#                                  here.
#   --use_kg_retrieval              Turns on the link module in _forward_stage2_batch.
#   --kg_rank 64 / --kg_hops 2 / --kg_damping 0.5   Defaults from the write-up
#                                  (E=512, r=64 worked example; hops=2 matches
#                                  HotpotQA/2WikiMultiHopQA's typical hop count).
#
# Everything else (data, epochs, LR, batch sizes, precision, checkpointing
# cadence) is left IDENTICAL to the baseline script on purpose.
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
    --stage2_retrieval_top_n 100
    --generation_top_k 5
    --use_kg_retrieval
    --kg_rank 64
    --kg_hops 2
    --kg_damping 0.5
    --qa_loss
    --do_eval_gen
)

# ─────────────────────────────────────────
# Launch
# ─────────────────────────────────────────
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --rdzv_id 104 \
    --rdzv_backend c10d \
    --rdzv_endpoint 127.0.0.1:29504 \
    --master_addr 127.0.0.1 \
    --master_port 29504 \
    --node_rank 0"

nvidia-smi
echo "Starting Stage 2 (end-to-end, KG-augmented) training — compression rate ${CR}x on ${NUM_GPUS} GPUs"
torchrun $DISTRIBUTED_ARGS -m "${training_commands[@]}"

# Sanity check: confirm the link module actually got saved, not just the usual files.
if [ -f "$SAVE_PATH/link_module.pth" ]; then
    echo "OK: link_module.pth found in $SAVE_PATH"
else
    echo "WARNING: link_module.pth NOT found in $SAVE_PATH -- KG weights were not persisted."
fi

# Copy model definition (and the new module) alongside checkpoint, matching
# the baseline script's existing convention.
cp $BASE_DIR/ml-clara/openrlhf/models/modeling_clara.py $SAVE_PATH
cp $BASE_DIR/ml-clara/openrlhf/models/temporal_link_matrix.py $SAVE_PATH

echo "Done!"