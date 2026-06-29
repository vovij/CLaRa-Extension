#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/latent_space_analysis_%j.log
#SBATCH --time=0-01:00:00
#SBATCH --job-name=clara_latent_space

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
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TMPDIR

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate
export PYTHONPATH=/vol/biomedic3/vd425/clara-main/ml-clara:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvcc --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
BASE_DIR=/vol/biomedic3/vd425/clara-main
DATASET=musique
N_DOCS=200
BATCH_SIZE=8
CRS="4 16 32 64 128 256"

OUT_DIR=$BASE_DIR/results/latent_space_analysis
mkdir -p $OUT_DIR $BASE_DIR/logs

# Baseline (stage1) checkpoints by default. To analyse the end-to-end model
# instead, override CKPT_ROOT, e.g.:
#   CKPT_ROOT="$BASE_DIR/checkpoints/CLaRa-7B-E2E/compression-{cr}"
CKPT_ROOT="$BASE_DIR/checkpoints/stage1/cr{cr}"

echo "Running latent space analysis"
echo "  dataset:   $DATASET"
echo "  n_docs:    $N_DOCS"
echo "  CRs:       $CRS"
echo "  ckpt_root: $CKPT_ROOT"
echo "  out_dir:   $OUT_DIR"

# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
python $BASE_DIR/scripts/eval/latent_space_analysis.py \
    --base_dir "$BASE_DIR" \
    --ckpt_root "$CKPT_ROOT" \
    --crs $CRS \
    --dataset "$DATASET" \
    --n_docs "$N_DOCS" \
    --batch_size "$BATCH_SIZE" \
    --out_dir "$OUT_DIR" \
    --device cuda

echo "Done! Results saved to $OUT_DIR"