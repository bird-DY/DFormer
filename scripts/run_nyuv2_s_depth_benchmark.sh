#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_nyuv2_s_depth_benchmark.sh --gpu GPU_ID --checkpoint PATH --report-root PATH [options]

Required:
  --gpu GPU_ID                Physical GPU index shown by nvidia-smi.
  --checkpoint PATH           DFormerv2-S NYUv2 checkpoint.
  --report-root PATH          Benchmark output directory.

Options:
  --protocol NAME             pilot, representative, core, or extended. Default: core.
  --checkpoint-label LABEL    Label stored in the manifest. Default: checkpoint.
  --python PATH               Python executable.
  --seed N                    Deterministic corruption seed. Default: 12345.
  --reserve-memory-mb N       GPU memory kept free. Default: 4096.
  --rerun                     Rerun completed conditions instead of resuming.
  --dry-run                   Validate and print all evaluation commands.
  -h, --help                  Show this message.

The benchmark is resumable. Re-running the same command skips conditions that
already contain a valid JSON report for the selected checkpoint.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=""
CHECKPOINT=""
REPORT_ROOT=""
PROTOCOL="core"
CHECKPOINT_LABEL="checkpoint"
PYTHON_BIN="/media/dell/Data1/newenv/dformer/bin/python"
CORRUPTION_SEED=12345
GPU_MEMORY_RESERVE_MB=4096
RERUN=0
DRY_RUN=0

while (($# > 0)); do
    case "$1" in
        --gpu) GPU_ID="${2:?Missing value for --gpu}"; shift 2 ;;
        --checkpoint) CHECKPOINT="${2:?Missing value for --checkpoint}"; shift 2 ;;
        --report-root) REPORT_ROOT="${2:?Missing value for --report-root}"; shift 2 ;;
        --protocol) PROTOCOL="${2:?Missing value for --protocol}"; shift 2 ;;
        --checkpoint-label) CHECKPOINT_LABEL="${2:?Missing value for --checkpoint-label}"; shift 2 ;;
        --python) PYTHON_BIN="${2:?Missing value for --python}"; shift 2 ;;
        --seed) CORRUPTION_SEED="${2:?Missing value for --seed}"; shift 2 ;;
        --reserve-memory-mb) GPU_MEMORY_RESERVE_MB="${2:?Missing value for --reserve-memory-mb}"; shift 2 ;;
        --rerun) RERUN=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$GPU_ID" || -z "$CHECKPOINT" || -z "$REPORT_ROOT" ]]; then
    echo "--gpu, --checkpoint, and --report-root are required." >&2
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
if [[ ! "$GPU_MEMORY_RESERVE_MB" =~ ^[0-9]+$ ]]; then
    echo "--reserve-memory-mb must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$PROTOCOL" =~ ^(pilot|representative|core|extended)$ ]]; then
    echo "--protocol must be pilot, representative, core, or extended; got: $PROTOCOL" >&2
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

command=(
    "${clean_env[@]}"
    "$PYTHON_BIN"
    -u scripts/run_depth_benchmark.py
    --checkpoint "$CHECKPOINT"
    --report-root "$REPORT_ROOT"
    --protocol "$PROTOCOL"
    --checkpoint-label "$CHECKPOINT_LABEL"
    --python "$PYTHON_BIN"
    --seed "$CORRUPTION_SEED"
    --reserve-memory-mb "$GPU_MEMORY_RESERVE_MB"
)
((RERUN == 1)) && command+=(--rerun)
((DRY_RUN == 1)) && command+=(--dry-run)

"${command[@]}"
