# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------
"""
evaluation.py

Goal (same flow as inference.ipynb):
1) Load config (YAML)
2) Build DataModule (optional; used here to fetch one sample like the notebook)
3) Build encoder -> network -> LightningModule
4) Forward one image:
     crops = model.window_imgs_semantic([img])
     mask_logits_per_layer, class_logits_per_layer = model(crops)
     mask_logits = interpolate(mask_logits_per_layer[-1], img_size)
     pixel_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_per_layer[-1])
     pixel_logits = model.revert_window_logits_semantic(pixel_logits, origins, img_sizes)[0]
5) From pixel_logits (C,H,W) compute anomaly maps:
     - MSP        : 1 - max softmax prob
     - MaxEntropy : entropy of softmax probs
     - MaxLogit   : -max channel value

No temperature scaling and no RbA (as requested).
"""

from __future__ import annotations

import argparse
import importlib
import warnings
from typing import Tuple, Optional

import yaml
import torch
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F
import numpy as np


# -----------------------------
# Helpers: dynamic import
# -----------------------------

def _import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# -----------------------------
# Build dataset (like inference.ipynb)
# -----------------------------

def build_data_from_config(config: dict, data_path: str):
    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_cls = getattr(importlib.import_module(data_module_name), class_name)
    data_kwargs = config["data"].get("init_args", {})

    # Mirror the notebook defaults for visualization/inference
    data = data_cls(
        path=data_path,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **data_kwargs,
    ).setup()
    return data


# -----------------------------
# Build model (encoder -> network -> lightning module)
# -----------------------------

def build_model_from_config(
    config: dict,
    img_size: Tuple[int, int],
    num_classes: int,
    device: torch.device,
    ckpt_path: Optional[str] = None,
    masked_attn_enabled: Optional[bool] = None,
):
    # Silence warning seen in the notebook
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    # --- encoder ---
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    EncoderCls = _import_class(encoder_cfg["class_path"])
    encoder_init = dict(encoder_cfg.get("init_args", {}))
    encoder = EncoderCls(img_size=img_size, **encoder_init)

    # --- network ---
    network_cfg = config["model"]["init_args"]["network"]
    NetworkCls = _import_class(network_cfg["class_path"])
    network_init = dict(network_cfg.get("init_args", {}))
    # encoder is passed explicitly
    network_init.pop("encoder", None)
    # allow overriding masked_attn_enabled from CLI
    if masked_attn_enabled is not None:
        network_init["masked_attn_enabled"] = masked_attn_enabled

    network = NetworkCls(
        num_classes=num_classes,
        encoder=encoder,
        **network_init,
    )

    # --- lightning module ---
    lit_cfg = config["model"]
    LitCls = _import_class(lit_cfg["class_path"])
    model_init = dict(lit_cfg.get("init_args", {}))
    model_init.pop("network", None)

    # Some tasks pass "stuff_classes" via data.init_args; handle if present
    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
        model_init["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = LitCls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_init,
    ).eval().to(device)

    # --- load weights (optional) ---
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # In some checkpoints, keys may include "._orig_mod" (compile); LightningModule strips in on_save_checkpoint,
        # but we still load permissively.
        model.load_state_dict(state, strict=False)

    return model


# -----------------------------
# Combine mask logits + class logits -> per-pixel logits (like inference.ipynb)
# -----------------------------

@torch.no_grad()
def forward_and_combine_semantic(
    model,
    img: torch.Tensor,
    device: torch.device,
    amp: bool = True,
):
    """
    Returns:
      pixel_logits: (C,H,W) on original image size
      mask_logits_raw: (1,Q,Hm,Wm) for the last layer BEFORE interpolation
      class_logits_raw: (1,Q,C+1) for the last layer
    """
    model.eval()

    # The project code expects uint8 [0,255] because it converts to PIL inside windowing.
    img_u8 = _ensure_uint8_0_255(img).to(device)

    imgs = [img_u8]
    img_sizes = [img_u8.shape[-2:] for img_u8 in imgs]

    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda", enabled=amp and device.type == "cuda"):
        crops, origins = model.window_imgs_semantic(imgs)  # (Ncrops,3,*,*)
        mask_logits_per_layer, class_logits_per_layer = model(crops)

        mask_logits_raw = mask_logits_per_layer[-1]
        class_logits_raw = class_logits_per_layer[-1]

        # Interpolate mask logits to model.img_size (same as inference.ipynb)
        mask_logits = F.interpolate(mask_logits_raw, model.img_size, mode="bilinear")

        # Combine (sigmoid masks, softmax classes) -> per-pixel "logits" (actually class score maps)
        crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_raw)  # (Ncrops,C,H,W)

        # Revert windowing back to original image resolution
        pixel_logits_list = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

    return pixel_logits_list[0], mask_logits_raw, class_logits_raw


def _ensure_uint8_0_255(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 3:
        raise ValueError(f"Expected img (3,H,W), got {tuple(img.shape)}")

    if img.dtype == torch.uint8:
        return img

    x = img.detach()

    # If it looks like [0,1], scale up
    if float(x.max()) <= 1.5:
        x = x * 255.0

    x = x.clamp(0, 255).to(torch.uint8)
    return x


# -----------------------------
# Anomaly scores (computed AFTER combine)
# -----------------------------

@torch.no_grad()
def anomaly_msp(pixel_logits: torch.Tensor) -> torch.Tensor:
    """
    MSP anomaly score:
      score(p) = 1 - max_c softmax(pixel_logits)_c
    """
    p = torch.softmax(pixel_logits, dim=0)
    return 1.0 - p.max(dim=0).values


@torch.no_grad()
def anomaly_maxentropy(pixel_logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    MaxEntropy anomaly score:
      score(p) = - sum_c p_c log p_c,  p = softmax(pixel_logits)
    (higher entropy => more anomaly)
    """
    p = torch.softmax(pixel_logits, dim=0)
    return -(p * (p + eps).log()).sum(dim=0)


@torch.no_grad()
def anomaly_maxlogit(pixel_logits: torch.Tensor) -> torch.Tensor:
    """
    MaxLogit anomaly score:
      score(p) = - max_c pixel_logits_c
    """
    return -pixel_logits.max(dim=0).values


# -----------------------------
# Metrics helpers (AUPRC, FPR@95)
# -----------------------------

def _to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().flatten().cpu().numpy()


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average Precision (AP) for binary labels. Returns a value in [0,1].

    Implementation matches the standard AP used in anomaly segmentation papers:
    sort by score descending, then sum precision at recall increments.
    """
    y_true = (y_true > 0).astype(np.uint8)

    pos = int(y_true.sum())
    if pos == 0:
        return float("nan")

    order = np.argsort(-scores)
    y = y_true[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos

    # AP = sum over i of (recall_i - recall_{i-1}) * precision_i
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - recall_prev) * precision))
    return ap


def fpr_at_95_tpr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """FPR@95TPR for binary labels. Returns a value in [0,1].

    Chooses threshold such that 95% of positive (anomaly) pixels are detected,
    then reports the false-positive rate on negative pixels.
    """
    y_true = (y_true > 0).astype(np.uint8)

    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]

    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")

    # threshold where TPR = 0.95 => keep top 95% positives => thr at 5th percentile
    thr = float(np.quantile(pos_scores, 0.05))
    fpr = float(np.mean(neg_scores >= thr))
    return fpr


def extract_gt_anomaly_and_ignore(
    target,
    out_hw: Tuple[int, int],
    gt_key: str = "anomaly_mask",
    ignore_value: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract a binary anomaly GT mask (1=anomaly, 0=normal) and an ignore mask.

    Supports common formats:
    - target is a torch.Tensor of shape (H,W)
    - target is a dict containing gt_key (H,W)

    Any pixels == ignore_value are excluded via ignore_mask.
    If GT size differs from out_hw, it is resized with nearest-neighbor.
    """
    H, W = out_hw

    if isinstance(target, torch.Tensor):
        gt = target
    elif isinstance(target, dict):
        if gt_key not in target:
            # Try a few common alternatives
            for k in ("anomaly", "gt_anomaly", "mask", "label", "labels", "anomaly_map"):
                if k in target:
                    gt = target[k]
                    break
            else:
                raise KeyError(
                    f"Could not find '{gt_key}' (or common alternatives) in target dict keys: {list(target.keys())}"
                )
        else:
            gt = target[gt_key]
    else:
        raise TypeError(f"Unsupported target type: {type(target)}")

    if not isinstance(gt, torch.Tensor):
        gt = torch.as_tensor(gt)

    if gt.dim() != 2:
        raise ValueError(f"Expected GT anomaly mask with shape (H,W), got {tuple(gt.shape)}")

    # Resize to output size if needed
    if gt.shape[-2:] != (H, W):
        gt_f = gt[None, None, ...].float()
        gt_rs = F.interpolate(gt_f, size=(H, W), mode="nearest")[0, 0]
        gt = gt_rs.to(gt.dtype)

    ignore_mask = gt == ignore_value
    # Common conventions: anomaly=1, normal=0. If your GT uses 255 for anomaly,
    # you can change this later, but for now we treat ignore_value as ignored.
    gt_bin = (gt > 0) & (~ignore_mask)

    return gt_bin.to(torch.uint8), ignore_mask


def compute_metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Returns (AUPRC%, FPR@95%)."""
    ap = average_precision(y_true, scores)  # [0,1]
    fpr = fpr_at_95_tpr(y_true, scores)     # [0,1]
    return ap * 100.0, fpr * 100.0


# -----------------------------
# Simple CLI demo (like inference.ipynb)
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="EoMT forward + combine + anomaly maps (like inference.ipynb).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/...yaml)")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset root directory (as used by the DataModule)")
    parser.add_argument("--img_idx", type=int, default=0, help="Index of image in data.val_dataloader().dataset")
    parser.add_argument(
        "--all",
        action="store_true",
        help="If set, run inference on the whole validation dataset (dataset[i]) instead of a single --img_idx.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap on number of validation images to process when using --all.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index in the validation dataset when using --all.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print a short progress line every N images when using --all.",
    )
    parser.add_argument(
        "--gt_key",
        type=str,
        default="anomaly_mask",
        help="Key to read the anomaly ground-truth mask from target dict (used for AUPRC/FPR@95).",
    )
    parser.add_argument(
        "--ignore_value",
        type=int,
        default=255,
        help="Ignore value in the GT anomaly mask (pixels == ignore_value are excluded).",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string (e.g. cuda:0 or cpu)")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to load weights")
    parser.add_argument("--amp", action="store_true", help="Enable torch autocast fp16 (recommended on CUDA)")
    parser.add_argument("--masked_attn_enabled", type=int, default=None,
                        help="Override network masked_attn_enabled (0/1). Default: use config/model default.")
    args = parser.parse_args()

    device = torch.device(args.device)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data = build_data_from_config(config, args.data_path)

    masked_attn_override = None
    if args.masked_attn_enabled is not None:
        masked_attn_override = bool(args.masked_attn_enabled)

    model = build_model_from_config(
        config=config,
        img_size=tuple(data.img_size),
        num_classes=int(data.num_classes),
        device=device,
        ckpt_path=args.ckpt,
        masked_attn_enabled=masked_attn_override,
    )

    val_dataset = data.val_dataloader().dataset

    if args.all:
        n_total = len(val_dataset)
        start = max(0, int(args.start_idx))
        if start >= n_total:
            raise ValueError(f"start_idx={start} is out of range for dataset of length {n_total}")

        if args.max_images is None:
            end = n_total
        else:
            end = min(n_total, start + int(args.max_images))

        print(f"Running on validation dataset indices [{start}, {end}) out of {n_total} images")

        # Accumulate flattened scores and labels for dataset-level metrics
        msp_scores, ment_scores, mlog_scores = [], [], []
        gt_labels = []

        for idx in range(start, end):
            img, target = val_dataset[idx]

            pixel_logits, mask_logits_raw, class_logits_raw = forward_and_combine_semantic(
                model=model,
                img=img,
                device=device,
                amp=args.amp,
            )

            H, W = int(pixel_logits.shape[-2]), int(pixel_logits.shape[-1])
            gt_bin, ignore_mask = extract_gt_anomaly_and_ignore(
                target,
                out_hw=(H, W),
                gt_key=args.gt_key,
                ignore_value=int(args.ignore_value),
            )

            # Compute anomaly maps
            msp = anomaly_msp(pixel_logits)
            ment = anomaly_maxentropy(pixel_logits)
            mlog = anomaly_maxlogit(pixel_logits)

            # Flatten and filter ignored pixels
            keep = ~ignore_mask
            y = gt_bin[keep]

            msp_scores.append(msp[keep])
            ment_scores.append(ment[keep])
            mlog_scores.append(mlog[keep])
            gt_labels.append(y)

            # Print occasionally (to avoid huge logs)
            if (idx == start) or ((idx - start) % int(args.print_every) == 0) or (idx == end - 1):
                print(
                    f"[{idx}] pixel_logits={tuple(pixel_logits.shape)} | "
                    f"kept_pixels={int(keep.sum().item())}"
                )

        # Concatenate everything
        y_all = _to_numpy_1d(torch.cat(gt_labels, dim=0))
        msp_all = _to_numpy_1d(torch.cat(msp_scores, dim=0))
        ment_all = _to_numpy_1d(torch.cat(ment_scores, dim=0))
        mlog_all = _to_numpy_1d(torch.cat(mlog_scores, dim=0))

        msp_auprc, msp_fpr95 = compute_metrics_from_scores(y_all, msp_all)
        ment_auprc, ment_fpr95 = compute_metrics_from_scores(y_all, ment_all)
        mlog_auprc, mlog_fpr95 = compute_metrics_from_scores(y_all, mlog_all)

        # Print as a simple table
        print("\n=== Dataset-level anomaly metrics (higher AUPRC is better, lower FPR@95 is better) ===")
        header = f"{'Method':<12} | {'AUPRC (%)':>10} | {'FPR@95 (%)':>10}"
        print(header)
        print("-" * len(header))
        print(f"{'MSP':<12} | {msp_auprc:10.2f} | {msp_fpr95:10.2f}")
        print(f"{'MaxEntropy':<12} | {ment_auprc:10.2f} | {ment_fpr95:10.2f}")
        print(f"{'MaxLogit':<12} | {mlog_auprc:10.2f} | {mlog_fpr95:10.2f}")

        print("Done.")

    else:
        # Fetch one sample like the notebook
        img, target = val_dataset[args.img_idx]

        pixel_logits, mask_logits_raw, class_logits_raw = forward_and_combine_semantic(
            model=model,
            img=img,
            device=device,
            amp=args.amp,
        )

        # Compute anomaly maps
        msp = anomaly_msp(pixel_logits).detach().cpu()
        ment = anomaly_maxentropy(pixel_logits).detach().cpu()
        mlog = anomaly_maxlogit(pixel_logits).detach().cpu()

        # Print shapes to confirm everything is consistent
        print("=== Forward outputs ===")
        print(f"mask_logits_raw:  {tuple(mask_logits_raw.shape)}  (last layer, before interpolate)")
        print(f"class_logits_raw: {tuple(class_logits_raw.shape)}  (last layer)")
        print(f"pixel_logits:     {tuple(pixel_logits.shape)}  (C,H,W on original resolution)")
        print("=== Anomaly maps ===")
        print(f"MSP:        {tuple(msp.shape)}  min={msp.min().item():.4f} max={msp.max().item():.4f}")
        print(f"MaxEntropy: {tuple(ment.shape)} min={ment.min().item():.4f} max={ment.max().item():.4f}")
        print(f"MaxLogit:   {tuple(mlog.shape)} min={mlog.min().item():.4f} max={mlog.max().item():.4f}")

    # Note: plotting is intentionally omitted here to keep the file minimal and dependency-free.
    # In Colab you can visualize msp/ment/mlog with matplotlib imshow if you want.


if __name__ == "__main__":
    main()
