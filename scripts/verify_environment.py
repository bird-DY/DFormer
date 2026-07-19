#!/usr/bin/env python3
"""Verify the pinned DFormer environment with a real GPU forward pass."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace


EXPECTED = {
    "python": (3, 10),
    "torch": "2.1.2",
    "torchvision": "0.16.2",
    "mmcv": "1.7.2",
    "mmengine": "0.10.4",
    "numpy": "1.26.4",
}


def clean_version(value: str) -> str:
    return value.split("+")[0]


def require_version(name: str, actual: str, expected: str) -> None:
    if clean_version(actual) != expected:
        raise RuntimeError(f"{name} version mismatch: expected {expected}, got {actual}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import mmcv
    import mmengine
    import numpy
    import torch
    import torchvision
    from mmcv.ops import get_compiler_version, get_compiling_cuda_version

    if sys.version_info[:2] != EXPECTED["python"]:
        raise RuntimeError(
            f"Python version mismatch: expected 3.10.x, got {platform.python_version()}"
        )

    require_version("PyTorch", torch.__version__, EXPECTED["torch"])
    require_version("torchvision", torchvision.__version__, EXPECTED["torchvision"])
    require_version("MMCV", mmcv.__version__, EXPECTED["mmcv"])
    require_version("MMEngine", mmengine.__version__, EXPECTED["mmengine"])
    require_version("NumPy", numpy.__version__, EXPECTED["numpy"])
    try:
        gui_opencv = importlib.metadata.version("opencv-python")
    except importlib.metadata.PackageNotFoundError:
        gui_opencv = None
    if gui_opencv is not None:
        raise RuntimeError(
            f"GUI OpenCV must not be installed on a headless server (found {gui_opencv})"
        )
    require_version(
        "opencv-python-headless",
        importlib.metadata.version("opencv-python-headless"),
        "4.9.0.80",
    )

    if torch.version.cuda != "11.8":
        raise RuntimeError(f"PyTorch must use CUDA 11.8, got {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check nvidia-smi and the host NVIDIA driver; "
            "installing a system CUDA toolkit is not required."
        )
    if os.name != "nt" and not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL is unavailable; distributed Linux training will not work")

    mmcv_cuda = get_compiling_cuda_version()
    if not mmcv_cuda.startswith("11.8"):
        raise RuntimeError(f"MMCV was not compiled for CUDA 11.8: got {mmcv_cuda}")

    # Import through the same paths used by the default DFormerv2 + Ham config.
    from models.builder import EncoderDecoder

    cfg = SimpleNamespace(
        backbone="DFormerv2_S",
        decoder="ham",
        decoder_embed_dim=512,
        num_classes=40,
        drop_path_rate=0.0,
        aux_rate=0.0,
        pretrained_model=None,
        bn_eps=1e-3,
        bn_momentum=0.1,
        background=255,
    )
    device = torch.device("cuda:0")
    model = EncoderDecoder(cfg=cfg, criterion=None, syncbn=False).eval().to(device)
    rgb = torch.randn(1, 3, 64, 64, device=device)
    depth = torch.randn(1, 3, 64, 64, device=device)
    with torch.inference_mode():
        output = model(rgb, depth)
    if tuple(output.shape) != (1, 40, 64, 64):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")

    print("Environment verification passed")
    print(f"  OS:           {platform.platform()}")
    print(f"  Python:       {platform.python_version()}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  Torch CUDA:   {torch.version.cuda}")
    print(f"  MMCV:         {mmcv.__version__}")
    print(f"  MMCV CUDA:    {mmcv_cuda}")
    print(f"  MMCV compiler:{get_compiler_version()}")
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
