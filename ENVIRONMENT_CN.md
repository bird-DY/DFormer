# DFormer / DFormerv2 服务器环境配置

## 推荐环境

这套版本以仓库当前代码实际使用的 API 为准：

| 组件 | 固定版本 | 原因 |
| --- | --- | --- |
| Python | 3.10.13 | 与官方 README 和预编译 wheel 匹配 |
| PyTorch | 2.1.2 | 代码使用 `torch.compile`、新版 AMP 和 `torchrun` |
| torchvision / torchaudio | 0.16.2 / 2.1.2 | 必须与 PyTorch 配套 |
| CUDA runtime | 11.8 | 由 PyTorch Conda 包提供，不要求服务器安装 CUDA Toolkit |
| mmcv-full | 1.7.2 | 仓库内置的 `mmseg 0.29.1` 要求 `mmcv>=1.3.13,<1.8.0` |
| MMEngine | 0.10.4 | 编码器和解码头直接导入其 checkpoint/BaseModule API |
| NumPy | 1.26.4 | 避免 PyTorch 2.1.x 与 NumPy 2.x 的二进制 ABI 冲突 |
| timm | 0.9.16 | DFormer/DFormerv2 使用 `DropPath` 和 `trunc_normal_` |

README 旧命令中的 `mmcv==2.1.0` 与仓库自带 `mmseg/__init__.py` 的版本检查冲突，默认的 DFormerv2 Ham 解码头会触发该检查。官方 OpenMMLab 的 CUDA 11.8 / PyTorch 2.1 wheel 索引同时提供了 `mmcv-full==1.7.2`，因此不需要源码编译。

## 服务器前置条件

- Linux x86_64 服务器和 NVIDIA GPU。
- `nvidia-smi` 能正常运行；驱动需要支持 CUDA 11.8。
- 已安装 Miniconda/Anaconda，并且当前 shell 可以找到 `conda`。
- 至少预留约 10 GB 环境空间。
- 安装阶段能够访问 PyPI、Anaconda 的 `defaults`/`pytorch`/`nvidia` 渠道和 `download.openmmlab.com`。

宿主机不需要单独安装 CUDA Toolkit、cuDNN、GCC 或 `nvcc`。脚本只接受预编译的 MMCV wheel；如果没有匹配 wheel，它会直接报错，而不会在服务器上静默编译。

脚本还会设置 `PYTHONNOUSERSITE=1`，避免服务器账户在 `~/.local` 中已有的包泄漏到新 Conda 环境。MMEngine 和 MMCV 虽然在元数据中要求 GUI 版 OpenCV，但脚本会确保最终只保留无图形界面服务器适用的 `opencv-python-headless`。

## 一键安装

在仓库根目录执行：

```bash
bash scripts/setup_server_env.sh --name dformer
conda activate dformer
```

脚本会依次安装固定版本、运行严格依赖检查，并用默认的 DFormerv2-S + Ham 解码头完成一次真实 GPU 前向测试。看到 `Environment verification passed` 才表示环境完整可用。

如果已经存在同名但依赖混乱的环境，使用全新环境名更安全：

```bash
bash scripts/setup_server_env.sh --name dformer-clean
```

只有确认旧环境不再需要时，才使用下面的重建命令（会删除同名 Conda 环境）：

```bash
bash scripts/setup_server_env.sh --name dformer --recreate
```

## 单独复查

```bash
conda activate dformer
python scripts/check_dependencies.py
python scripts/verify_environment.py
```

然后检查数据集和权重路径：

```text
datasets/NYUDepthv2/{RGB,Depth,Label,train.txt,test.txt}
checkpoints/pretrained/DFormerv2_Small_pretrained.pth
```

路径正确后才运行 `bash train.sh`。当前 `train.sh` 默认使用 2 张 GPU；单卡服务器需要同时把 `GPUS`、`CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node` 调整为 1。

## 常见故障

### `MMCV==2.1.0 is used but incompatible`

说明装了 README 旧版本。不要同时安装 `mmcv`、`mmcv-lite` 和 `mmcv-full`。重新运行一键脚本，或创建一个干净环境。

### `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

说明 NumPy 被升级了。执行：

```bash
python -m pip install --force-reinstall numpy==1.26.4
```

### MMCV 下载了 `.tar.gz` 或开始执行 `setup.py`

这表示 Python、PyTorch、CUDA 或 CPU 架构与 wheel 不匹配。一键脚本使用 `--only-binary mmcv-full`，会在这种情况下立即失败。不要继续源码编译；确认服务器是 Linux x86_64、Python 3.10、PyTorch 2.1.x、CUDA 11.8 组合。

### `libGL.so.1` 缺失

服务器装到了 GUI 版 OpenCV。此方案固定使用 `opencv-python-headless`；移除 `opencv-python` 后重新运行脚本。

`mmengine` 和 `mmcv-full` 的元数据固定写着 `opencv-python`，所以原始 `pip check` 会把 headless 版误报为缺失。仓库的 `scripts/check_dependencies.py` 仅在 `cv2` 与 `opencv-python-headless` 版本完全一致时豁免这两条 OpenMMLab 告警，其他依赖冲突仍会导致失败。

### `torch.cuda.is_available()` 为 `False`

先运行 `nvidia-smi`。如果该命令也失败，需要管理员修复 NVIDIA 驱动或容器 GPU 映射；反复安装 CUDA Toolkit 不能修复驱动问题。

### `No module named timm` / `No module named mmengine`

这是原 README 漏列的直接依赖。不要逐个临时安装，重新执行一键脚本以恢复完整固定版本。

## 依赖范围说明

`requirements/runtime.txt` 覆盖训练、评估、可视化、FLOPs 统计所需包。`utils/demo_geometry_prior.py` 还引用了仓库中不存在的 `grid_gen`，因此该独立演示脚本不属于环境验收范围；它不影响 DFormerv2 训练、评估和推理。
