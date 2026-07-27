#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_nyuv2_s_inference_matrix.sh --gpu GPU_ID --reproduced-checkpoint PATH [options]

Required:
  --gpu GPU_ID                    Physical GPU index shown by nvidia-smi.
  --reproduced-checkpoint PATH    Best checkpoint from the single-GPU reproduction.

Options:
  --author-checkpoint PATH        Author checkpoint. Default:
                                  checkpoints/trained/NYU Depth v2/DFormerv2_Small_NYU.pth
  --report-root PATH              Output directory for this run.
  --python PATH                   Python executable.
  --reserve-memory-mb N           Keep this much GPU memory free. Default: 4096.
  --core-only                     Run only A and D for both weights (4 core evaluations).
  --force                         Re-run strategies that already have a JSON report.
  -h, --help                      Show this message.

The full matrix contains:
  A_ss_noflip: scale 1.0, no flip
  B_ss_flip:   scale 1.0, horizontal flip TTA
  C_ms_noflip: scales 0.5,0.75,1.0,1.25,1.5, no flip
  D_ms_flip:   the five scales plus horizontal flip TTA

All strategies use clean depth, sliding-window inference, AMP, and validation batch size 1.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=""
PYTHON_BIN="/media/dell/Data1/newenv/dformer/bin/python"
AUTHOR_CHECKPOINT="checkpoints/trained/NYU Depth v2/DFormerv2_Small_NYU.pth"
REPRODUCED_CHECKPOINT=""
REPORT_ROOT=""
GPU_MEMORY_RESERVE_MB=4096
CORE_ONLY=0
FORCE=0

while (($# > 0)); do
    case "$1" in
        --gpu)
            [[ $# -ge 2 ]] || { echo "Missing value for --gpu" >&2; exit 2; }
            GPU_ID="$2"
            shift 2
            ;;
        --author-checkpoint)
            [[ $# -ge 2 ]] || { echo "Missing value for --author-checkpoint" >&2; exit 2; }
            AUTHOR_CHECKPOINT="$2"
            shift 2
            ;;
        --reproduced-checkpoint)
            [[ $# -ge 2 ]] || { echo "Missing value for --reproduced-checkpoint" >&2; exit 2; }
            REPRODUCED_CHECKPOINT="$2"
            shift 2
            ;;
        --report-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --report-root" >&2; exit 2; }
            REPORT_ROOT="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --reserve-memory-mb)
            [[ $# -ge 2 ]] || { echo "Missing value for --reserve-memory-mb" >&2; exit 2; }
            GPU_MEMORY_RESERVE_MB="$2"
            shift 2
            ;;
        --core-only)
            CORE_ONLY=1
            shift
            ;;
        --force)
            FORCE=1
            shift
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

if [[ -z "$GPU_ID" || -z "$REPRODUCED_CHECKPOINT" ]]; then
    echo "--gpu and --reproduced-checkpoint are required." >&2
    usage
    exit 2
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "--gpu must be a non-negative integer, got: $GPU_ID" >&2
    exit 2
fi
if [[ ! "$GPU_MEMORY_RESERVE_MB" =~ ^[0-9]+$ ]]; then
    echo "--reserve-memory-mb must be a non-negative integer" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$AUTHOR_CHECKPOINT" ]]; then
    echo "Author checkpoint not found: $AUTHOR_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$REPRODUCED_CHECKPOINT" ]]; then
    echo "Reproduced checkpoint not found: $REPRODUCED_CHECKPOINT" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    free_gpu_memory="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
    free_gpu_memory="$(printf '%s\n' "$free_gpu_memory" | head -n 1 | tr -d '[:space:]')"
    if [[ "$free_gpu_memory" =~ ^[0-9]+$ ]] && ((free_gpu_memory <= GPU_MEMORY_RESERVE_MB)); then
        echo "GPU $GPU_ID has ${free_gpu_memory} MiB free, not enough for a ${GPU_MEMORY_RESERVE_MB} MiB reserve." >&2
        exit 1
    fi
    if [[ "$free_gpu_memory" =~ ^[0-9]+$ ]]; then
        echo "GPU $GPU_ID: ${free_gpu_memory} MiB free; reserving ${GPU_MEMORY_RESERVE_MB} MiB."
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
report_root="${REPORT_ROOT:-validation_reports/inference_matrix_nyuv2_s_${run_id}}"
mkdir -p "$report_root"

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "gpu_id=$GPU_ID"
    echo "author_checkpoint=$AUTHOR_CHECKPOINT"
    echo "reproduced_checkpoint=$REPRODUCED_CHECKPOINT"
    echo "reserve_memory_mb=$GPU_MEMORY_RESERVE_MB"
    echo "sliding=true"
    echo "amp=true"
} > "$report_root/experiment.meta"

if ((CORE_ONLY)); then
    strategies=(A_ss_noflip D_ms_flip)
else
    strategies=(A_ss_noflip B_ss_flip C_ms_noflip D_ms_flip)
fi

run_eval() {
    local weight_name="$1"
    local checkpoint="$2"
    local strategy="$3"
    local strategy_dir="$report_root/$weight_name/$strategy"
    local flip_arg
    local scales

    case "$strategy" in
        A_ss_noflip)
            flip_arg="--no-flip"
            scales=(1.0)
            ;;
        B_ss_flip)
            flip_arg="--flip"
            scales=(1.0)
            ;;
        C_ms_noflip)
            flip_arg="--no-flip"
            scales=(0.5 0.75 1.0 1.25 1.5)
            ;;
        D_ms_flip)
            flip_arg="--flip"
            scales=(0.5 0.75 1.0 1.25 1.5)
            ;;
        *)
            echo "Unknown strategy: $strategy" >&2
            return 2
            ;;
    esac

    mkdir -p "$strategy_dir"
    if ((FORCE == 0)) && compgen -G "$strategy_dir/*.json" >/dev/null; then
        echo "Skipping completed evaluation: $weight_name/$strategy"
        return
    fi

    echo
    echo "===== Evaluating $weight_name/$strategy ====="
    "${clean_env[@]}" "$PYTHON_BIN" -u utils/eval.py \
        --config=local_configs.NYUDepthv2.DFormerv2_S \
        --gpus=1 \
        --val_batch_size=1 \
        --gpu_memory_reserve_mb="$GPU_MEMORY_RESERVE_MB" \
        --no-syncbn \
        --amp \
        --no-compile \
        --no-mst \
        --eval_scales "${scales[@]}" \
        "$flip_arg" \
        --sliding \
        --depth_corruption=clean \
        --continue_fpath="$checkpoint" \
        --report_dir="$strategy_dir" \
        2>&1 | tee "$strategy_dir/eval.log"
}

for weight_spec in "author|$AUTHOR_CHECKPOINT" "reproduced|$REPRODUCED_CHECKPOINT"; do
    weight_name="${weight_spec%%|*}"
    checkpoint="${weight_spec#*|}"
    for strategy in "${strategies[@]}"; do
        run_eval "$weight_name" "$checkpoint" "$strategy"
    done
done

summary_args=()
if ((CORE_ONLY)); then
    summary_args+=(--allow-incomplete)
fi
"${clean_env[@]}" "$PYTHON_BIN" scripts/summarize_inference_matrix.py "$report_root" "${summary_args[@]}"
echo "Inference matrix completed: $REPO_ROOT/$report_root"
