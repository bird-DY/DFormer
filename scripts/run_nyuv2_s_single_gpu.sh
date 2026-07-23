#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_nyuv2_s_single_gpu.sh --gpu GPU_ID [options]

Required:
  --gpu GPU_ID                Physical GPU index shown by nvidia-smi.

Options:
  --python PATH               Python executable.
  --micro-batch-size N        Default: 8.
  --grad-accum-steps N        Default: 2.
  --val-batch-size N          Default: 1.
  -h, --help                  Show this message.

Example:
  bash scripts/run_nyuv2_s_single_gpu.sh --gpu 0
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=""
PYTHON_BIN="/media/dell/Data1/newenv/dformer/bin/python"
MICRO_BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
VAL_BATCH_SIZE=1

while (($# > 0)); do
    case "$1" in
        --gpu)
            [[ $# -ge 2 ]] || { echo "Missing value for --gpu" >&2; usage; exit 2; }
            GPU_ID="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; usage; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --micro-batch-size)
            [[ $# -ge 2 ]] || { echo "Missing value for --micro-batch-size" >&2; usage; exit 2; }
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum-steps)
            [[ $# -ge 2 ]] || { echo "Missing value for --grad-accum-steps" >&2; usage; exit 2; }
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --val-batch-size)
            [[ $# -ge 2 ]] || { echo "Missing value for --val-batch-size" >&2; usage; exit 2; }
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

if [[ -z "$GPU_ID" ]]; then
    echo "--gpu is required; no GPU is selected implicitly." >&2
    usage
    exit 2
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "--gpu must be a non-negative integer, got: $GPU_ID" >&2
    exit 2
fi
for value_name in MICRO_BATCH_SIZE GRAD_ACCUM_STEPS VAL_BATCH_SIZE; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer, got: $value" >&2
        exit 2
    fi
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ $((MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS)) -ne 16 ]]; then
    echo "micro batch size times gradient accumulation steps must equal 16" >&2
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
echo "Selected physical GPU: $GPU_ID"
"${clean_env[@]}" "$PYTHON_BIN" -c '
import torch
from mmcv.ops import get_compiling_cuda_version

if torch.version.cuda is None:
    raise RuntimeError("A CPU-only PyTorch build is installed")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the selected CUDA device")

print("Torch path:", torch.__file__)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("MMCV CUDA:", get_compiling_cuda_version())
print("Visible GPU:", torch.cuda.get_device_name(0))
'

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

nohup "${clean_env[@]}" \
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

echo "Training started on physical GPU $GPU_ID"
echo "PID: $train_pid"
echo "Log: $REPO_ROOT/$log_file"
echo "PID file: $REPO_ROOT/$pid_file"
echo "Metadata: $REPO_ROOT/$meta_file"
echo "Monitor: tail -f '$REPO_ROOT/$log_file'"
