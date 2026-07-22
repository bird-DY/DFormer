#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/media/dell/Data1/newenv/dformer/bin/python}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ $((MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS)) -ne 16 ]]; then
    echo "MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS must equal the configured batch size 16" >&2
    exit 1
fi

for required_file in \
    datasets/NYUDepthv2/train.txt \
    datasets/NYUDepthv2/test.txt \
    checkpoints/pretrained/DFormerv2_Small_pretrained.pth; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $REPO_ROOT/$required_file" >&2
        exit 1
    fi
done

mkdir -p run_logs
run_id="$(date +%Y%m%d-%H%M%S)"
log_file="run_logs/A0_clean_dformerv2_s_gpu${GPU_ID}_${run_id}.log"
pid_file="run_logs/A0_clean_dformerv2_s_gpu${GPU_ID}_${run_id}.pid"
meta_file="run_logs/A0_clean_dformerv2_s_gpu${GPU_ID}_${run_id}.meta"

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "gpu_id=$GPU_ID"
    echo "micro_batch_size=$MICRO_BATCH_SIZE"
    echo "grad_accum_steps=$GRAD_ACCUM_STEPS"
    echo "effective_batch_size=$((MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS))"
    echo "val_batch_size=$VAL_BATCH_SIZE"
    echo "python=$PYTHON_BIN"
} > "$meta_file"

nohup env \
    -u PYTHONHOME \
    -u CUDA_HOME \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    LOCAL_RANK=0 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT" \
    "$PYTHON_BIN" -u utils/train.py \
    --config=local_configs.NYUDepthv2.DFormerv2_S \
    --gpus=1 \
    --micro_batch_size="$MICRO_BATCH_SIZE" \
    --grad_accum_steps="$GRAD_ACCUM_STEPS" \
    --val_batch_size="$VAL_BATCH_SIZE" \
    --no-syncbn \
    --amp \
    --val_amp \
    --no-compile \
    --no-mst \
    --no-sliding \
    --use_seed \
    > "$log_file" 2>&1 &

train_pid=$!
echo "$train_pid" > "$pid_file"

echo "Training started"
echo "PID: $train_pid"
echo "Log: $REPO_ROOT/$log_file"
echo "PID file: $REPO_ROOT/$pid_file"
echo "Metadata: $REPO_ROOT/$meta_file"
echo "Monitor: tail -f '$REPO_ROOT/$log_file'"
