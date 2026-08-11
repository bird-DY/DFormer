import argparse
import csv
import json
import pprint
import re
import time
from importlib import import_module
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tabulate import tabulate
from models.builder import EncoderDecoder as segmodel
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from val_mm import evaluate, evaluate_msf

from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.depth_corruptions import DEPTH_CORRUPTIONS, build_depth_corruptor
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger

# from eval import evaluate_mid

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--eval_scales",
    nargs="+",
    type=float,
    default=None,
    help=(
        "evaluation scales, for example --eval_scales 1.0 or "
        "--eval_scales 0.5 0.75 1.0 1.25 1.5; overrides the scale choice implied by --mst"
    ),
)
parser.add_argument(
    "--flip",
    default=None,
    action=argparse.BooleanOptionalAction,
    help="enable horizontal-flip test-time augmentation; overrides the flip choice implied by --mst",
)
parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--pad_SUNRGBD", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--val_batch_size", default=1, type=int, help="validation batch size")
parser.add_argument(
    "--gpu_memory_reserve_mb",
    default=0,
    type=int,
    help="leave this much GPU memory unallocated by the evaluation process (best effort)",
)
parser.add_argument(
    "--depth_corruption",
    default="clean",
    choices=DEPTH_CORRUPTIONS,
    help="deterministic corruption applied only to validation depth images before normalization",
)
parser.add_argument("--corruption_seed", default=12345, type=int)
parser.add_argument("--missing_rate", default=0.3, type=float)
parser.add_argument(
    "--noise_std",
    default=0.03,
    type=float,
    help="Gaussian noise standard deviation relative to the 8-bit depth range",
)
parser.add_argument("--shift_x", default=4, type=int, help="horizontal depth translation in pixels")
parser.add_argument("--shift_y", default=0, type=int, help="vertical depth translation in pixels")
parser.add_argument("--block_size", default=128, type=int, help="square missing-region size in pixels")
parser.add_argument("--block_count", default=1, type=int, help="number of square missing regions")
parser.add_argument("--blur_kernel", default=5, type=int, help="odd Gaussian-blur kernel size")
parser.add_argument("--depth_scale", default=1.0, type=float, help="multiplicative depth-scale factor")
parser.add_argument("--outlier_rate", default=0.03, type=float, help="fraction of valid pixels replaced by random depth")
parser.add_argument(
    "--condition_id",
    default=None,
    help="stable benchmark condition identifier; defaults to the corruption name",
)
parser.add_argument(
    "--benchmark_protocol",
    default="custom",
    help="benchmark protocol recorded in JSON reports",
)
parser.add_argument(
    "--report_dir",
    default="validation_reports",
    help="directory used to save validation CSV and JSON reports",
)
# parser.add_argument('--save_path', '-p', default=None)

# os.environ['MASTER_PORT'] = '169710'
torch.set_float32_matmul_precision("high")
import torch._dynamo

torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.automatic_dynamic_shapes = False


def _safe_percent(numerator, denominator):
    result = torch.zeros_like(numerator, dtype=torch.float64)
    valid = denominator > 0
    result[valid] = numerator[valid].double() / denominator[valid].double() * 100.0
    return result


def report_metrics(metric, config, model, args, elapsed_seconds, num_images):
    """Print and persist a complete semantic-segmentation validation report."""
    hist = metric.hist.detach().double().cpu()
    true_positive = hist.diag()
    ground_truth = hist.sum(dim=1)
    predicted = hist.sum(dim=0)
    union = ground_truth + predicted - true_positive

    iou = _safe_percent(true_positive, union)
    precision = _safe_percent(true_positive, predicted)
    recall = _safe_percent(true_positive, ground_truth)
    f1 = _safe_percent(2.0 * true_positive, ground_truth + predicted)

    valid_pixels = ground_truth.sum().item()
    pixel_acc = 100.0 * true_positive.sum().item() / valid_pixels if valid_pixels else 0.0
    frequency = ground_truth / valid_pixels if valid_pixels else torch.zeros_like(ground_truth)
    fw_iou = (frequency * iou).sum().item()

    miou = iou.mean().item()
    mprecision = precision.mean().item()
    macc = recall.mean().item()
    mf1 = f1.mean().item()

    class_names = list(getattr(config, "class_names", []))
    if len(class_names) != config.num_classes:
        class_names = [f"class_{index}" for index in range(config.num_classes)]

    rows = []
    per_class = []
    for index, class_name in enumerate(class_names):
        values = {
            "class_id": index,
            "class_name": class_name,
            "iou_percent": round(iou[index].item(), 2),
            "precision_percent": round(precision[index].item(), 2),
            "recall_percent": round(recall[index].item(), 2),
            "f1_percent": round(f1[index].item(), 2),
            "support_pixels": int(ground_truth[index].item()),
        }
        per_class.append(values)
        rows.append(
            [
                values["class_id"],
                values["class_name"],
                values["iou_percent"],
                values["precision_percent"],
                values["recall_percent"],
                values["f1_percent"],
                values["support_pixels"],
            ]
        )

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    fps = num_images / elapsed_seconds if elapsed_seconds > 0 else 0.0
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )

    condition_id = args.condition_id or args.depth_corruption
    summary = {
        "dataset": config.dataset_name,
        "model": config.backbone,
        "checkpoint": str(args.continue_fpath),
        "condition_id": condition_id,
        "benchmark_protocol": args.benchmark_protocol,
        "depth_corruption": args.depth_corruption,
        "corruption_seed": int(args.corruption_seed),
        "missing_rate": float(args.missing_rate),
        "noise_std": float(args.noise_std),
        "shift_x": int(args.shift_x),
        "shift_y": int(args.shift_y),
        "block_size": int(args.block_size),
        "block_count": int(args.block_count),
        "blur_kernel": int(args.blur_kernel),
        "depth_scale": float(args.depth_scale),
        "outlier_rate": float(args.outlier_rate),
        "num_images": int(num_images),
        "num_classes": int(config.num_classes),
        "eval_scales": [float(scale) for scale in args.eval_scales],
        "multi_scale": len(args.eval_scales) > 1,
        "flip": bool(args.flip),
        "multi_scale_flip": len(args.eval_scales) > 1 and bool(args.flip),
        "sliding_window": bool(args.sliding),
        "amp": bool(args.amp),
        "miou_percent": round(miou, 2),
        "mean_precision_percent": round(mprecision, 2),
        "mean_accuracy_percent": round(macc, 2),
        "mean_f1_percent": round(mf1, 2),
        "pixel_accuracy_percent": round(pixel_acc, 2),
        "frequency_weighted_iou_percent": round(fw_iou, 2),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "images_per_second": round(fps, 3),
        "peak_gpu_memory_mb": round(peak_gpu_memory_mb, 2),
        "gpu_memory_total_mb": int(args.gpu_memory_total_mb),
        "gpu_memory_free_at_start_mb": int(args.gpu_memory_free_at_start_mb),
        "gpu_memory_budget_mb": int(args.gpu_memory_budget_mb),
        "gpu_memory_reserve_mb": int(args.gpu_memory_reserve_mb),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
    }

    print("\n=== Validation summary ===", flush=True)
    print(
        tabulate(
            [[key, value] for key, value in summary.items()],
            headers=["Metric", "Value"],
            tablefmt="github",
        ),
        flush=True,
    )
    print("\n=== Per-class metrics ===", flush=True)
    print(
        tabulate(
            rows,
            headers=["ID", "Class", "IoU (%)", "Precision (%)", "Recall/Acc (%)", "F1 (%)", "Support"],
            tablefmt="github",
            floatfmt=".2f",
        ),
        flush=True,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    safe_condition_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", condition_id)
    report_stem = f"{config.dataset_name}_{config.backbone}_{safe_condition_id}_{timestamp}"
    json_path = report_dir / f"{report_stem}.json"
    csv_path = report_dir / f"{report_stem}.csv"

    with json_path.open("w", encoding="utf-8") as report_file:
        json.dump(
            {
                "summary": summary,
                "per_class": per_class,
                "confusion_matrix": hist.long().tolist(),
            },
            report_file,
            ensure_ascii=False,
            indent=2,
        )

    with csv_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(per_class)

    print(f"\nJSON report: {json_path.resolve()}", flush=True)
    print(f"CSV report:  {csv_path.resolve()}", flush=True)
    return summary

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")
    if args.val_batch_size < 1:
        raise ValueError("--val_batch_size must be at least 1")
    if args.gpu_memory_reserve_mb < 0:
        raise ValueError("--gpu_memory_reserve_mb must be non-negative")
    # Keep --mst/--no-mst backward compatible while allowing scale and flip
    # test-time augmentation to be controlled independently.
    if args.eval_scales is None:
        args.eval_scales = [0.5, 0.75, 1.0, 1.25, 1.5] if args.mst else [1.0]
    if any(scale <= 0 for scale in args.eval_scales):
        raise ValueError("--eval_scales values must all be greater than zero")
    if args.flip is None:
        args.flip = bool(args.mst)
    args.use_msf = len(args.eval_scales) > 1 or args.flip
    depth_corruption = build_depth_corruptor(args)
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    # check if pad_SUNRGBD is used correctly
    if args.pad_SUNRGBD and config.dataset_name != "SUNRGBD":
        args.pad_SUNRGBD = False
        logger.warning("pad_SUNRGBD is only used for SUNRGBD dataset")
    if (args.pad_SUNRGBD) and (not config.backbone.startswith("DFormerv2")):
        raise ValueError("DFormerv1 is not recommended with pad_SUNRGBD")
    if (not args.pad_SUNRGBD) and config.backbone.startswith("DFormerv2") and config.dataset_name == "SUNRGBD":
        raise ValueError("DFormerv2 is not recommended without pad_SUNRGBD")
    config.pad = args.pad_SUNRGBD

    args.gpu_memory_total_mb = 0
    args.gpu_memory_free_at_start_mb = 0
    args.gpu_memory_budget_mb = 0
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        reserve_bytes = args.gpu_memory_reserve_mb * 1024**2
        budget_bytes = free_bytes - reserve_bytes
        if budget_bytes <= 0:
            raise RuntimeError(
                f"GPU has {free_bytes / 1024**2:.0f} MiB free, which is not greater than "
                f"the requested {args.gpu_memory_reserve_mb} MiB reserve"
            )
        if args.gpu_memory_reserve_mb > 0:
            memory_fraction = min(budget_bytes / total_bytes, 1.0)
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)

        args.gpu_memory_total_mb = total_bytes // 1024**2
        args.gpu_memory_free_at_start_mb = free_bytes // 1024**2
        args.gpu_memory_budget_mb = budget_bytes // 1024**2
        logger.info(
            "GPU memory at evaluation start: total=%d MiB, free=%d MiB, budget=%d MiB, reserve=%d MiB",
            args.gpu_memory_total_mb,
            args.gpu_memory_free_at_start_mb,
            args.gpu_memory_budget_mb,
            args.gpu_memory_reserve_mb,
        )

    cudnn.benchmark = True
    val_batch_size = args.val_batch_size

    if args.mst:
        val_loader, val_sampler = get_val_loader(
            engine,
            RGBXDataset,
            config,
            val_batch_size=val_batch_size,
            depth_corruption=depth_corruption,
        )
    else:
        val_loader, val_sampler = get_val_loader(
            engine,
            RGBXDataset,
            config,
            val_batch_size=val_batch_size,
            depth_corruption=depth_corruption,
        )
    logger.info(f"val dataset len:{len(val_loader) * int(args.gpus)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n" + pp.pformat(config))

    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)

    if args.syncbn:
        BatchNorm2d = nn.SyncBatchNorm
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )

    weight = torch.load(args.continue_fpath, map_location=torch.device("cpu"))
    if "model" in weight:
        weight = weight["model"]
    elif "state_dict" in weight:
        weight = weight["state_dict"]

    logger.info(f"load model from {args.continue_fpath}")
    print(model.load_state_dict(weight, strict=False))

    if engine.distributed:
        logger.info(".............distributed training.............")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=True,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    if args.compile:
        model = torch.compile(model, backend="inductor", mode=args.compile_mode)

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    eval_start_time = time.perf_counter()
    num_val_images = len(val_loader.dataset)
    if args.amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.use_msf:
                        all_metrics = evaluate_msf(
                            model,
                            val_loader,
                            config,
                            device,
                            args.eval_scales,
                            args.flip,
                            engine,
                            sliding=args.sliding,
                        )
                    else:
                        all_metrics = evaluate(
                            model,
                            val_loader,
                            config,
                            device,
                            engine,
                            sliding=args.sliding,
                        )
                    if engine.local_rank == 0:
                        metric = all_metrics[0]
                        for other_metric in all_metrics[1:]:
                            metric.update_hist(other_metric.hist)
                        report_metrics(
                            metric,
                            config,
                            model,
                            args,
                            time.perf_counter() - eval_start_time,
                            num_val_images,
                        )
            elif not engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.use_msf:
                        metric = evaluate_msf(
                            model,
                            val_loader,
                            config,
                            device,
                            args.eval_scales,
                            args.flip,
                            engine,
                            sliding=args.sliding,
                        )
                    else:
                        metric = evaluate(
                            model,
                            val_loader,
                            config,
                            device,
                            engine,
                            sliding=args.sliding,
                        )
                    report_metrics(
                        metric,
                        config,
                        model,
                        args,
                        time.perf_counter() - eval_start_time,
                        num_val_images,
                    )
    else:
        if engine.distributed:
            with torch.no_grad():
                model.eval()
                device = torch.device("cuda")
                if args.use_msf:
                    all_metrics = evaluate_msf(
                        model,
                        val_loader,
                        config,
                        device,
                        args.eval_scales,
                        args.flip,
                        engine,
                        sliding=args.sliding,
                    )
                else:
                    all_metrics = evaluate(
                        model,
                        val_loader,
                        config,
                        device,
                        engine,
                        sliding=args.sliding,
                    )
                if engine.local_rank == 0:
                    metric = all_metrics[0]
                    for other_metric in all_metrics[1:]:
                        metric.update_hist(other_metric.hist)
                    report_metrics(
                        metric,
                        config,
                        model,
                        args,
                        time.perf_counter() - eval_start_time,
                        num_val_images,
                    )
        elif not engine.distributed:
            with torch.no_grad():
                model.eval()
                device = torch.device("cuda")
                if args.use_msf:
                    metric = evaluate_msf(
                        model,
                        val_loader,
                        config,
                        device,
                        args.eval_scales,
                        args.flip,
                        engine,
                        sliding=args.sliding,
                    )
                else:
                    metric = evaluate(
                        model,
                        val_loader,
                        config,
                        device,
                        engine,
                        sliding=args.sliding,
                    )
                report_metrics(
                    metric,
                    config,
                    model,
                    args,
                    time.perf_counter() - eval_start_time,
                    num_val_images,
                )
    logger.info("end testing")
    print("end testing", flush=True)
