#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_depth_robustness_nyuv2_s.sh --gpu GPU_ID [options]

Required:
  --gpu GPU_ID                Physical GPU index shown by nvidia-smi.

Options:
  --checkpoint PATH           Checkpoint to evaluate.
  --report-root PATH          Output directory for this five-condition run.
  --python PATH               Python executable.
  --seed N                    Corruption seed. Default: 12345.
  --max-initial-memory-mb N   Refuse a GPU using more than this. Default: 2048.
  -h, --help                  Show this message.

Examples:
  bash scripts/run_depth_robustness_nyuv2_s.sh --gpu 0
  bash scripts/run_depth_robustness_nyuv2_s.sh --gpu 1 --checkpoint /path/to/best.pth
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=""
PYTHON_BIN="/media/dell/Data1/newenv/dformer/bin/python"
CHECKPOINT="checkpoints/trained/NYU Depth v2/DFormerv2_Small_NYU.pth"
REPORT_ROOT=""
MAX_INITIAL_GPU_MEMORY_MB=2048
CORRUPTION_SEED=12345

while (($# > 0)); do
    case "$1" in
        --gpu)
            [[ $# -ge 2 ]] || { echo "Missing value for --gpu" >&2; usage; exit 2; }
            GPU_ID="$2"
            shift 2
            ;;
        --checkpoint)
            [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint" >&2; usage; exit 2; }
            CHECKPOINT="$2"
            shift 2
            ;;
        --report-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --report-root" >&2; usage; exit 2; }
            REPORT_ROOT="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; usage; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; usage; exit 2; }
            CORRUPTION_SEED="$2"
            shift 2
            ;;
        --max-initial-memory-mb)
            [[ $# -ge 2 ]] || { echo "Missing value for --max-initial-memory-mb" >&2; usage; exit 2; }
            MAX_INITIAL_GPU_MEMORY_MB="$2"
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
if [[ ! "$CORRUPTION_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "--seed must be an integer, got: $CORRUPTION_SEED" >&2
    exit 2
fi
if [[ ! "$MAX_INITIAL_GPU_MEMORY_MB" =~ ^[0-9]+$ ]]; then
    echo "--max-initial-memory-mb must be a non-negative integer" >&2
    exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $REPO_ROOT/$CHECKPOINT" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    initial_gpu_memory="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
    initial_gpu_memory="$(printf '%s\n' "$initial_gpu_memory" | head -n 1 | tr -d '[:space:]')"
    if [[ "$initial_gpu_memory" =~ ^[0-9]+$ ]] && ((initial_gpu_memory > MAX_INITIAL_GPU_MEMORY_MB)); then
        echo "GPU $GPU_ID already uses ${initial_gpu_memory} MiB; refusing to compete with the active job." >&2
        echo "Wait for it to become idle, pass another --gpu value, or explicitly raise --max-initial-memory-mb." >&2
        exit 1
    fi
fi

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

run_id="$(date +%Y%m%d-%H%M%S)"
report_root="${REPORT_ROOT:-validation_reports/depth_robustness_author_s_${run_id}}"
mkdir -p "$report_root"

run_eval() {
    local condition="$1"
    shift
    local condition_dir="$report_root/$condition"
    mkdir -p "$condition_dir"
    echo
    echo "===== Evaluating $condition ====="
    "${clean_env[@]}" "$PYTHON_BIN" -u utils/eval.py \
        --config=local_configs.NYUDepthv2.DFormerv2_S \
        --gpus=1 \
        --val_batch_size=1 \
        --no-syncbn \
        --amp \
        --no-compile \
        --no-mst \
        --no-sliding \
        --corruption_seed="$CORRUPTION_SEED" \
        --continue_fpath="$CHECKPOINT" \
        --report_dir="$condition_dir" \
        "$@" \
        2>&1 | tee "$condition_dir/eval.log"
}

run_eval clean --depth_corruption=clean
run_eval random_missing --depth_corruption=random_missing --missing_rate=0.30
run_eval gaussian_noise --depth_corruption=gaussian_noise --noise_std=0.03
run_eval shift --depth_corruption=shift --shift_x=4 --shift_y=0
run_eval zero --depth_corruption=zero

"${clean_env[@]}" "$PYTHON_BIN" scripts/summarize_depth_robustness.py "$report_root"
echo "Depth robustness benchmark completed: $REPO_ROOT/$report_root"
