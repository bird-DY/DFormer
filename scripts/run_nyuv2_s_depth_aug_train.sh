#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_nyuv2_s_depth_aug_train.sh --gpu GPU_ID [options]

Required:
  --gpu GPU_ID                Physical GPU index shown by nvidia-smi.

Options:
  --seed N                    Default: 12345.
  --corruption-probability P  Default: 0.5.
  --python PATH               Python executable.
  -h, --help                  Show this message.

This A1 baseline keeps half of the training samples clean. The other half
uniformly samples one of: random missing 30/50%, Gaussian noise 0.03/0.05,
horizontal shift 4/8 px, or complete depth removal. RGB and labels stay clean.
The DFormerv2-S optimizer batch remains 16 (micro batch 8, accumulation 2).
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=""
SEED=12345
CORRUPTION_PROBABILITY=0.5
PYTHON_BIN="/media/dell/Data1/newenv/dformer/bin/python"

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
        --corruption-probability)
            [[ $# -ge 2 ]] || { echo "Missing value for --corruption-probability" >&2; exit 2; }
            CORRUPTION_PROBABILITY="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 2; }
            PYTHON_BIN="$2"
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

probability_tag="${CORRUPTION_PROBABILITY/./}"
RUN_LABEL="depthaug_s_p${probability_tag}"

echo "Starting A1: DFormerv2-S with representative depth corruption augmentation"
echo "Physical GPU:           $GPU_ID"
echo "Seed:                   $SEED"
echo "Corruption probability: $CORRUPTION_PROBABILITY"
echo "Clean probability:      computed as 1 - corruption probability"
echo "Run label:               $RUN_LABEL"

bash scripts/run_nyuv2_single_gpu.sh \
    --gpu "$GPU_ID" \
    --variant S \
    --seed "$SEED" \
    --python "$PYTHON_BIN" \
    --micro-batch-size 8 \
    --grad-accum-steps 2 \
    --val-batch-size 1 \
    --run-label "$RUN_LABEL" \
    --train-depth-augmentation \
    --train-depth-corruption-probability "$CORRUPTION_PROBABILITY" \
    --train-depth-corruption-profile representative
