#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="dformer"
RECREATE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_server_env.sh [--name ENV_NAME] [--recreate]

Create a reproducible DFormer/DFormerv2 CUDA 11.8 environment and run a GPU
smoke test. Existing environments are updated in place unless --recreate is
specified.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            [[ $# -ge 2 ]] || { echo "error: --name requires a value" >&2; exit 2; }
            ENV_NAME="$2"
            shift 2
            ;;
        --recreate)
            RECREATE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_REQUIREMENTS="${REPO_ROOT}/requirements/runtime.txt"
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html"

# A Conda environment must not accidentally satisfy dependencies from the
# account's user-level site-packages directory.
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

command -v conda >/dev/null 2>&1 || {
    echo "error: conda was not found. Install Miniconda, reopen the shell, and retry." >&2
    exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "error: nvidia-smi was not found. Install a CUDA 11.8-compatible NVIDIA driver first." >&2
    exit 1
}
[[ -f "${RUNTIME_REQUIREMENTS}" ]] || {
    echo "error: ${RUNTIME_REQUIREMENTS} is missing" >&2
    exit 1
}

eval "$(conda shell.bash hook)"

env_exists() {
    conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq -- "$ENV_NAME"
}

if [[ "$RECREATE" -eq 1 ]] && env_exists; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda deactivate >/dev/null 2>&1 || true
    conda env remove --name "$ENV_NAME" --yes
fi

if ! env_exists; then
    echo "Creating environment: ${ENV_NAME}"
    conda create --name "$ENV_NAME" --yes \
        python=3.10.13 pip=24.0 setuptools=69.5.1 wheel=0.43.0
else
    echo "Updating existing environment: ${ENV_NAME}"
    conda install --name "$ENV_NAME" --yes \
        python=3.10.13 pip=24.0 setuptools=69.5.1 wheel=0.43.0
fi

run_in_env() {
    conda run --no-capture-output --name "$ENV_NAME" "$@"
}

echo "Installing the pinned PyTorch CUDA 11.8 build..."
conda install --name "$ENV_NAME" --yes \
    pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    pytorch-cuda=11.8 -c pytorch -c nvidia

echo "Installing pinned Python dependencies..."
run_in_env python -m pip install --requirement "$RUNTIME_REQUIREMENTS"

echo "Installing MMEngine without its GUI OpenCV dependency..."
run_in_env python -m pip install mmengine==0.10.4 --no-deps

# Both MMEngine and mmcv-full declare opencv-python even though the headless
# distribution provides the same cv2 module. Keep only the server-safe build.
run_in_env python -m pip uninstall --yes opencv-python >/dev/null 2>&1 || true
run_in_env python -m pip install \
    opencv-python-headless==4.9.0.80 --force-reinstall --no-deps

echo "Installing the prebuilt MMCV wheel (source builds are intentionally disabled)..."
run_in_env python -m pip uninstall --yes mmcv mmcv-lite mmcv-full >/dev/null 2>&1 || true
run_in_env python -m pip install \
    mmcv-full==1.7.2 \
    --find-links "$MMCV_INDEX" \
    --only-binary mmcv-full \
    --no-deps

echo "Checking package metadata..."
run_in_env python "${REPO_ROOT}/scripts/check_dependencies.py"

echo "Running the DFormerv2 GPU smoke test..."
cd "$REPO_ROOT"
run_in_env python scripts/verify_environment.py

cat <<EOF

DFormer environment is ready.

Activate it with:
  conda activate ${ENV_NAME}

Then review dataset/checkpoint paths in local_configs/ before training.
EOF
