#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_nyuv2_s_batch_ablation.sh --gpu GPU_ID [options]

Required:
  --gpu GPU_ID                Physical GPU index shown by nvidia-smi.

Options:
  --seed N                    Default: 12345 (paired with the reproduced baseline).
  --statistics-batch-size N   One of 8, 4, 2, or 1. Default: 4.
  --python PATH               Python executable passed to the training launcher.
  --val-batch-size N          Default: 1.
  -h, --help                  Show this message.

The optimizer batch is fixed at 16. Gradient accumulation is selected as
16 / statistics-batch-size, so this experiment changes the normalization
statistics batch without changing the optimizer batch, images per epoch,
optimizer updates per epoch, learning-rate schedule, or random seed.

Recommended first run on GPU 0:
  bash scripts/run_nyuv2_s_batch_ablation.sh --gpu 0 --seed 12345 --statistics-batch-size 4
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=""
SEED=12345
STATISTICS_BATCH_SIZE=4
PYTHON_BIN="/media/dell/Data1/newenv/dformer/bin/python"
VAL_BATCH_SIZE=1

while (($# > 0)); do
    case "$1" in
        --gpu)
            [[ $# -ge 2 ]] || { echo "Missing value for --gpu" >&2; exit 2; }
            GPU_ID="$2"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 2; }
            SEED="$2"
            shift 2
            ;;
        --statistics-batch-size)
            [[ $# -ge 2 ]] || { echo "Missing value for --statistics-batch-size" >&2; exit 2; }
            STATISTICS_BATCH_SIZE="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 2; }
            PYTHON_BIN="$2"
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

if [[ -z "$GPU_ID" ]]; then
    echo "--gpu is required; no GPU is selected implicitly." >&2
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
case "$STATISTICS_BATCH_SIZE" in
    8|4|2|1) ;;
    *)
        echo "--statistics-batch-size must be one of 8, 4, 2, or 1; got: $STATISTICS_BATCH_SIZE" >&2
        exit 2
        ;;
esac

EFFECTIVE_BATCH_SIZE=16
GRAD_ACCUM_STEPS=$((EFFECTIVE_BATCH_SIZE / STATISTICS_BATCH_SIZE))
RUN_LABEL="nobd_s_stat${STATISTICS_BATCH_SIZE}_opt${EFFECTIVE_BATCH_SIZE}"

echo "Starting the NYUv2 DFormerv2-S normalization/optimization batch ablation"
echo "Physical GPU:                 $GPU_ID"
echo "Seed:                         $SEED"
echo "Normalization statistics:     $STATISTICS_BATCH_SIZE"
echo "Gradient accumulation steps:  $GRAD_ACCUM_STEPS"
echo "Effective optimizer batch:    $EFFECTIVE_BATCH_SIZE"
echo "Run label:                     $RUN_LABEL"

bash scripts/run_nyuv2_single_gpu.sh \
    --gpu "$GPU_ID" \
    --variant S \
    --seed "$SEED" \
    --python "$PYTHON_BIN" \
    --micro-batch-size "$STATISTICS_BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --run-label "$RUN_LABEL"
