#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_nyuv2_s_depth_benchmark_pair.sh --gpu GPU_ID --reproduced-checkpoint PATH [options]

Required:
  --gpu GPU_ID                     Physical GPU index.
  --reproduced-checkpoint PATH     Reproduced DFormerv2-S NYUv2 checkpoint.

Options:
  --author-checkpoint PATH         Default: checkpoints/trained/NYU Depth v2/DFormerv2_Small_NYU.pth
  --report-root PATH               Parent output directory.
  --protocol NAME                  pilot, representative, core, or extended. Default: core.
  --python PATH                    Python executable.
  --seed N                         Corruption seed. Default: 12345.
  --reserve-memory-mb N            GPU memory kept free. Default: 4096.
  --rerun                          Rerun completed conditions.
  --dry-run                        Print commands without evaluation.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=""
AUTHOR_CHECKPOINT="checkpoints/trained/NYU Depth v2/DFormerv2_Small_NYU.pth"
REPRODUCED_CHECKPOINT=""
REPORT_ROOT=""
PROTOCOL="core"
PYTHON_BIN="/media/dell/Data1/newenv/dformer/bin/python"
CORRUPTION_SEED=12345
GPU_MEMORY_RESERVE_MB=4096
RERUN=0
DRY_RUN=0

while (($# > 0)); do
    case "$1" in
        --gpu) GPU_ID="${2:?Missing value for --gpu}"; shift 2 ;;
        --author-checkpoint) AUTHOR_CHECKPOINT="${2:?Missing value for --author-checkpoint}"; shift 2 ;;
        --reproduced-checkpoint) REPRODUCED_CHECKPOINT="${2:?Missing value for --reproduced-checkpoint}"; shift 2 ;;
        --report-root) REPORT_ROOT="${2:?Missing value for --report-root}"; shift 2 ;;
        --protocol) PROTOCOL="${2:?Missing value for --protocol}"; shift 2 ;;
        --python) PYTHON_BIN="${2:?Missing value for --python}"; shift 2 ;;
        --seed) CORRUPTION_SEED="${2:?Missing value for --seed}"; shift 2 ;;
        --reserve-memory-mb) GPU_MEMORY_RESERVE_MB="${2:?Missing value for --reserve-memory-mb}"; shift 2 ;;
        --rerun) RERUN=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$GPU_ID" || -z "$REPRODUCED_CHECKPOINT" ]]; then
    echo "--gpu and --reproduced-checkpoint are required." >&2
    usage
    exit 2
fi

run_id="$(date +%Y%m%d-%H%M%S)"
REPORT_ROOT="${REPORT_ROOT:-validation_reports/depth_benchmark_pair_s_${PROTOCOL}_${run_id}}"
common_args=(
    --gpu "$GPU_ID"
    --protocol "$PROTOCOL"
    --python "$PYTHON_BIN"
    --seed "$CORRUPTION_SEED"
    --reserve-memory-mb "$GPU_MEMORY_RESERVE_MB"
)
((RERUN == 1)) && common_args+=(--rerun)
((DRY_RUN == 1)) && common_args+=(--dry-run)

bash scripts/run_nyuv2_s_depth_benchmark.sh \
    "${common_args[@]}" \
    --checkpoint "$AUTHOR_CHECKPOINT" \
    --checkpoint-label author \
    --report-root "$REPORT_ROOT/author"

bash scripts/run_nyuv2_s_depth_benchmark.sh \
    "${common_args[@]}" \
    --checkpoint "$REPRODUCED_CHECKPOINT" \
    --checkpoint-label reproduced \
    --report-root "$REPORT_ROOT/reproduced"

if ((DRY_RUN == 0)); then
    env \
        -u PYTHONHOME \
        -u CUDA_HOME \
        -u LD_LIBRARY_PATH \
        -u LD_PRELOAD \
        PYTHONNOUSERSITE=1 \
        "PYTHONPATH=$REPO_ROOT" \
        "$PYTHON_BIN" scripts/compare_depth_benchmarks.py \
        --reference-root "$REPORT_ROOT/author" \
        --candidate-root "$REPORT_ROOT/reproduced" \
        --reference-label author \
        --candidate-label reproduced \
        --output-dir "$REPORT_ROOT/comparison"
fi

echo "Depth benchmark pair completed: $REPO_ROOT/$REPORT_ROOT"
