#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:4
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/eval_stage2_cr4_kg_%j.log
#SBATCH --time=23:59:59
#SBATCH --job-name=eval_stage2_cr4_kg

set -ex

source /vol/cuda/12.9.0/setup.sh
export CUDA_HOME=/vol/cuda/12.9.0
export PATH=$CUDA_HOME/bin:$PATH
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME=/vol/biomedic3/vd425/.cache/huggingface
export HF_DATASETS_CACHE=/vol/biomedic3/vd425/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/vol/biomedic3/vd425/.cache/triton
export TORCH_EXTENSIONS_DIR=/vol/biomedic3/vd425/.cache/torch_extensions
export TMPDIR=/vol/biomedic3/vd425/.tmp
export MPLCONFIGDIR=/vol/biomedic3/vd425/.cache/matplotlib
export PYTHONUNBUFFERED=1
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TMPDIR $MPLCONFIGDIR

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate

BASE_DIR=/vol/biomedic3/vd425/clara-main
CKPT_PATH=$BASE_DIR/checkpoints/stage2_kg/cr4

unset PYTHONPATH
export PYTHONPATH=$BASE_DIR/ml-clara:$CKPT_PATH

# One-line pre-flight check, cheap and fast (no GPU touched), confirming
# use_kg_retrieval actually made it into this checkpoint's config.json
# before spending a 4-GPU allocation on it.
python -c "import json; cfg = json.load(open('$CKPT_PATH/config.json')); assert cfg.get('use_kg_retrieval') is True, f'use_kg_retrieval={cfg.get(\"use_kg_retrieval\")} -- aborting'; print('OK: KG config confirmed, kg_rank=%s kg_hops=%s kg_damping=%s' % (cfg.get('kg_rank'), cfg.get('kg_hops'), cfg.get('kg_damping')))"

accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    $BASE_DIR/ml-clara/evaluation/evaluate.py \
        --model_path  "$CKPT_PATH" \
        --stage       stage2 \
        --dataset     nq,hotpotqa,musique,2wiki \
        --generation_top_k 5 \
        --batch_size  4 \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}