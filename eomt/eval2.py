
import os
import glob
import random
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve

from lightning.pytorch.cli import LightningCLI

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule


# -------------------------
# GT mapping (ONLY change requested)
# -------------------------
def map_gt_to_ood_binary(ood_gts: np.ndarray, dataset: str) -> np.ndarray:
    """
    Returns mask with:
      0   = in-distribution
      1   = OOD/anomaly
      255 = ignore/void
    Based on YOUR inspect:
      - RoadAnomaly: {0,2}
      - RoadAnomaly21/RoadObsticle21/LostAndFound/fs_static: {0,1,255}
    """
    ds = dataset.lower()
    u = set(np.unique(ood_gts).tolist())

    def as_uint8(x):
        return x.astype(np.uint8, copy=False)

    # RoadAnomaly: {0,2} -> map 2->1
    if "roadanomaly" in ds and "roadanomaly21" not in ds:
        out = np.zeros_like(ood_gts, dtype=np.uint8)
        out[ood_gts == 2] = 1
        valid = (ood_gts == 0) | (ood_gts == 2)
        out[~valid] = 255
        return out

    # All others you listed: already {0,1,255}
    if (
        "roadanomaly21" in ds
        or "roadobsticle21" in ds or "roadobstacle21" in ds
        or "lostandfound" in ds or "lost&found" in ds or "lost_and_found" in ds
        or "fs_static" in ds or "fishyscapes" in ds
    ):
        if u.issubset({0, 1, 255}):
            return as_uint8(ood_gts)
        raise ValueError(f"[{dataset}] Unexpected GT values: {sorted(u)} (expected subset of {{0,1,255}})")

    raise ValueError(f"Unknown dataset: {dataset}")


# -------------------------
# metrics utils
# -------------------------
def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0
    return float(fpr[idxs[0]])


def parse_list_floats(s: str):
    s = s.replace(",", " ").strip()
    return [float(x) for x in s.split() if x.strip()]


def parse_list_str(s: str):
    s = s.replace(",", " ").strip()
    return [x.strip() for x in s.split() if x.strip()]


def normalize_probs(per_pixel_scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    per_pixel_scores: [B,C,H,W] (not guaranteed to sum to 1 across C because of mask overlap)
    """
    denom = per_pixel_scores.sum(dim=1, keepdim=True).clamp_min(eps)
    return per_pixel_scores / denom


def anomaly_map_from_scores(per_pixel_scores_chw: torch.Tensor, method: str, eps: float = 1e-8) -> torch.Tensor:
    """
    per_pixel_scores_chw: [C,H,W] non-negative "scores" per class per pixel
    method:
      - msp: 1 - max(prob)
      - maxentropy: entropy(prob)
      - maxlogit: -max(score)   (best-effort since we don't have true per-pixel logits)
    """
    method = method.lower()

    # convert scores->probs for MSP/entropy
    probs = per_pixel_scores_chw / per_pixel_scores_chw.sum(dim=0, keepdim=True).clamp_min(eps)

    if method == "msp":
        msp = probs.max(dim=0).values
        return 1.0 - msp

    if method == "maxentropy":
        entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=0)
        return entropy

    if method == "maxlogit":
        # not true logits; we use negative max score as a simple confidence proxy
        maxs = per_pixel_scores_chw.max(dim=0).values
        return -maxs

    raise ValueError(f"Unsupported method: {method}")


# -------------------------
# LightningCLI wrapper (run=False)
# -------------------------
class EvalOnlyCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, run=False, save_config_callback=None, **kwargs)


# -------------------------
# helpers
# -------------------------
def infer_dataset_from_path(path_like: str) -> str:
    p = path_like.lower()
    if "roadanomaly21" in p:
        return "RoadAnomaly21"
    if "roadobsticle21" in p or "roadobstacle21" in p:
        return "RoadObsticle21"
    if "lostandfound" in p or "lost&found" in p or "lost_and_found" in p:
        return "LostAndFound"
    if "fs_static" in p or "fishyscapes" in p:
        return "fs_static"
    if "roadanomaly" in p:
        return "RoadAnomaly"
    raise ValueError(f"Cannot infer dataset from path: {path_like}")


def default_gt_path_from_img(path_img: str) -> str:
    # Keep your original convention: images -> labels_masks
    pathGT = path_img.replace("images", "labels_masks")

    p = pathGT.lower()

    # Your original dataset-specific extension fixes (done safely)
    if ("roadobsticle21" in p or "roadobstacle21" in p) and pathGT.lower().endswith(".webp"):
        pathGT = pathGT[:-5] + ".png"
    if ("fs_static" in p or "fishyscapes" in p) and pathGT.lower().endswith(".jpg"):
        pathGT = pathGT[:-4] + ".png"
    if "roadanomaly" in p and pathGT.lower().endswith(".jpg"):
        pathGT = pathGT[:-4] + ".png"

    return pathGT


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .bin or weights file used by model.init_args.ckpt_path")
    parser.add_argument("--input", type=str, required=True, help='Glob images, e.g. "/path/RoadAnomaly/images/*.png"')
    parser.add_argument("--temps", type=str, default="1.0", help='List of temperatures, e.g. "0.5,1,2,4"')
    parser.add_argument("--methods", type=str, default="msp", help='List of methods: "msp,maxlogit,maxentropy"')
    parser.add_argument("--img_size", type=int, nargs=2, default=None, help="Override H W at runtime (optional)")
    parser.add_argument("--cpu", type=int, default=0, help="1 to force CPU")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu == 1 or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    temps = parse_list_floats(args.temps)
    methods = [m.lower() for m in parse_list_str(args.methods)]
    print("Temps:", temps)
    print("Methods:", methods)

    # Build project LightningModule from YAML (NO fit)
    # IMPORTANT: disable logger here to avoid W&B init during eval
    cli_args = [
        "--config", args.config,
        "--model.init_args.ckpt_path", args.ckpt_path,
        "--trainer.logger=false",
        "--trainer.enable_checkpointing=false",
    ]

    cli = EvalOnlyCLI(
    LightningModule,
    LightningDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    args=cli_args,   # <-- QUI la differenza: non parser_kwargs
)

    lm: LightningModule = cli.model
    lm.to(device)
    lm.eval()

    if args.img_size is not None:
        lm.img_size = tuple(args.img_size)

    IMG_SIZE = tuple(lm.img_size)  # (H,W)
    print("IMG_SIZE used:", IMG_SIZE)

    file_list = sorted(glob.glob(os.path.expanduser(args.input)))
    print(f"Found {len(file_list)} images")

    # Collect per (T, method)
    scores = defaultdict(list)  # key=(T,method) -> list of anomaly maps (H,W)
    gts = defaultdict(list)     # key=(T,method) -> list of GT masks (H,W)

    for path in file_list:
        # --- load image as 0..255 float (LightningModule.forward divides by 255 internally)
        img_pil = Image.open(path).convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
        img_np = np.array(img_pil, dtype=np.uint8)  # H,W,3
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)  # [1,3,H,W]

        # forward (same as project)
        mask_logits_layers, class_logits_layers = lm(img_tensor)
        mask_logits = mask_logits_layers[-1]    # [B,Q,h,w]
        class_logits = class_logits_layers[-1]  # [B,Q,C+1]

        # upsample masks to IMG_SIZE
        mask_logits = F.interpolate(mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False)

        # --- GT
        pathGT = default_gt_path_from_img(path)
        if not os.path.exists(pathGT):
            continue

        gt_img = Image.open(pathGT).resize((IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST)
        ood_gts_raw = np.array(gt_img)

        ds = infer_dataset_from_path(pathGT)
        ood_gts_bin = map_gt_to_ood_binary(ood_gts_raw, ds)

        # Skip if there is no OOD pixel
        if not np.any(ood_gts_bin == 1):
            continue

        # --- temperature sweep
        for T in temps:
            clsT = class_logits / float(T)

            # per-pixel class "scores" [B,C,H,W]
            per_pixel_scores = LightningModule.to_per_pixel_logits_semantic(mask_logits, clsT)

            # Normalize to avoid scale drift (still keep scores for maxlogit if you want)
            per_pixel_scores = normalize_probs(per_pixel_scores)  # [B,C,H,W]
            scores_chw = per_pixel_scores[0]  # [C,H,W]

            for method in methods:
                amap = anomaly_map_from_scores(scores_chw, method)
                key = (float(T), method)
                scores[key].append(amap.detach().cpu().numpy())
                gts[key].append(ood_gts_bin)

    if not scores:
        print("No valid samples collected (missing GT or no OOD pixels).")
        return

    # --- compute metrics per key
    print("\n================ RESULTS ================")
    for key in sorted(scores.keys(), key=lambda x: (x[0], x[1])):
        T, method = key
        anomaly_scores = np.stack(scores[key], axis=0)  # [N,H,W]
        ood_masks = np.stack(gts[key], axis=0)          # [N,H,W]

        valid = (ood_masks != 255)
        labels = ood_masks[valid].astype(np.uint8)      # 0/1
        sc = anomaly_scores[valid].astype(np.float32)

        # Need both classes
        if labels.max() == labels.min():
            print(f"T={T:g} method={method:10s} -> skipped (only one class present)")
            continue

        auprc = average_precision_score(labels, sc)
        fpr95 = fpr_at_95_tpr(sc, labels)

        print(f"T={T:g}  method={method:10s} | AUPRC={auprc*100:6.2f}% | FPR@95TPR={fpr95*100:6.2f}%")
    print("========================================\n")


if __name__ == "__main__":
    main()