#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_nyuv2_single_gpu.sh --gpu GPU_ID --variant S|B|L --seed N [options]

Required:
  --gpu GPU_ID                Physical GPU index shown by nvidia-smi.
  --variant S|B|L             DFormerv2 model size.
  --seed N                    Training random seed.

Options:
  --python PATH               Python executable.
  --micro-batch-size N        Override the conservative per-GPU default.
  --grad-accum-steps N        Override the accumulation default.
  --val-batch-size N          Default: 1.
  -h, --help                  Show this message.

Defaults for a 24 GiB RTX A5000:
  S: micro batch 8, accumulation 2, effective batch 16
  B: micro batch 4, accumulation 2, effective batch 8
  L: micro batch 2, accumulation 6, effective batch 12

If B or L runs out of memory, retry with the same effective batch:
  B: --micro-batch-size 2 --grad-accum-steps 4
  L: --micro-batch-size 1 --grad-accum-steps 12
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=""
VARIANT=""
SEED=""
PYTHON_BIN="/media/dell/Data1/newenv/dformer/bin/python"
MICRO_BATCH_SIZE=""
GRAD_ACCUM_STEPS=""
VAL_BATCH_SIZE=1

while (($# > 0)); do
    case "$1" in
        --gpu)
            [[ $# -ge 2 ]] || { echo "Missing value for --gpu" >&2; exit 2; }
            GPU_ID="$2"
            shift 2
            ;;
        --variant)
            [[ $# -ge 2 ]] || { echo "Missing value for --variant" >&2; exit 2; }
            VARIANT="${2^^}"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 2; }
            SEED="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --micro-batch-size)
            [[ $# -ge 2 ]] || { echo "Missing value for --micro-batch-size" >&2; exit 2; }
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum-steps)
            [[ $# -ge 2 ]] || { echo "Missing value for --grad-accum-steps" >&2; exit 2; }
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --val-batch-size)
            [[ $# -ge 2 ]] || { echo "Missing value for --val-batch-size" >&2; exit 2; }
            VAL_BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$GPU_ID" || -z "$VARIANT" || -z "$SEED" ]]; then
    echo "--gpu, --variant, and --seed are required." >&2
    usage
    exit 2
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "--gpu must be a non-negative integer, got: $GPU_ID" >&2
    exit 2
fi
if [[ ! "$SEED" =~ ^-?[0-9]+$ ]]; then
    echo "--seed must be an integer, got: $SEED" >&2
    exit 2
fi

case "$VARIANT" in
    S)
        CONFIG="local_configs.NYUDepthv2.DFormerv2_S"
        PRETRAINED="checkpoints/pretrained/DFormerv2_Small_pretrained.pth"
        CONFIGURED_BATCH_SIZE=16
        DEFAULT_MICRO_BATCH_SIZE=8
        DEFAULT_GRAD_ACCUM_STEPS=2
        ;;
    B)
        CONFIG="local_configs.NYUDepthv2.DFormerv2_B"
        PRETRAINED="checkpoints/pretrained/DFormerv2_Base_pretrained.pth"
        CONFIGURED_BATCH_SIZE=8
        DEFAULT_MICRO_BATCH_SIZE=4
        DEFAULT_GRAD_ACCUM_STEPS=2
        ;;
    L)
        CONFIG="local_configs.NYUDepthv2.DFormerv2_L"
        PRETRAINED="checkpoints/pretrained/DFormerv2_Large_pretrained.pth"
        CONFIGURED_BATCH_SIZE=12
        DEFAULT_MICRO_BATCH_SIZE=2
        DEFAULT_GRAD_ACCUM_STEPS=6
        ;;
    *)
        echo "--variant must be one of S, B, or L; got: $VARIANT" >&2
        exit 2
        ;;
esac

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-$DEFAULT_MICRO_BATCH_SIZE}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$DEFAULT_GRAD_ACCUM_STEPS}"

for value_name in MICRO_BATCH_SIZE GRAD_ACCUM_STEPS VAL_BATCH_SIZE; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer, got: $value" >&2
        exit 2
    fi
done
if [[ $((MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS)) -ne "$CONFIGURED_BATCH_SIZE" ]]; then
    echo "micro batch size times gradient accumulation steps must equal $CONFIGURED_BATCH_SIZE for DFormerv2-$VARIANT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

for required_file in \
    datasets/NYUDepthv2/train.txt \
    datasets/NYUDepthv2/test.txt \
    "$PRETRAINED"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $REPO_ROOT/$required_file" >&2
        exit 1
    fi
done

clean_env=(
    env
    -u PYTHONHOME
    -u CUDA_HOME
    -u LD_LIBRARY_PATH
    -u LD_PRELOAD
    "CUDA_VISIBLE_DEVICES=$GPU_ID"
    LOCAL_RANK=0
    PYTHONNOUSERSITE=1
    PYTHONUNBUFFERED=1
    "PYTHONPATH=$REPO_ROOT"
)

echo "Checking physical GPU $GPU_ID with the isolated PyTorch/MMCV CUDA runtime..."
"${clean_env[@]}" "$PYTHON_BIN" -c '
import torch
from mmcv.ops import get_compiling_cuda_version

if torch.version.cuda is None:
    raise RuntimeError("A CPU-only PyTorch build is installed")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the selected CUDA device")

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("MMCV CUDA:", get_compiling_cuda_version())
print("Visible GPU:", torch.cuda.get_device_name(0))
'

mkdir -p run_logs
run_id="$(date +%Y%m%d-%H%M%S)"
variant_lower="${VARIANT,,}"
log_file="run_logs/nyuv2_dformerv2_${variant_lower}_seed${SEED}_gpu${GPU_ID}_${run_id}.log"
pid_file="run_logs/nyuv2_dformerv2_${variant_lower}_seed${SEED}_gpu${GPU_ID}_${run_id}.pid"
meta_file="run_logs/nyuv2_dformerv2_${variant_lower}_seed${SEED}_gpu${GPU_ID}_${run_id}.meta"

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "variant=$VARIANT"
    echo "config=$CONFIG"
    echo "seed=$SEED"
    echo "gpu_id=$GPU_ID"
    echo "micro_batch_size=$MICRO_BATCH_SIZE"
    echo "grad_accum_steps=$GRAD_ACCUM_STEPS"
    echo "effective_batch_size=$((MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS))"
    echo "val_batch_size=$VAL_BATCH_SIZE"
    echo "pretrained=$PRETRAINED"
    echo "python=$PYTHON_BIN"
} > "$meta_file"

nohup "${clean_env[@]}" \
    "$PYTHON_BIN" -u utils/train.py \
    --config="$CONFIG" \
    --gpus=1 \
    --seed="$SEED" \
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

echo "DFormerv2-$VARIANT training started on physical GPU $GPU_ID with seed $SEED"
echo "PID: $train_pid"
echo "Log: $REPO_ROOT/$log_file"
echo "PID file: $REPO_ROOT/$pid_file"
echo "Metadata: $REPO_ROOT/$meta_file"
echo "Monitor: tail -f '$REPO_ROOT/$log_file'"
