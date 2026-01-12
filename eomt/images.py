# export_best_image.py
# Seleziona la "best image" (massimo AP per-image) e salva una figura 2x2:
# Input / GT(OOD mask) / Prediction(semantic argmax) / Anomaly score
#
# Uso:
# python export_best_image.py \
#   --input "/path/images/*.jpg" \
#   --loadDir "/path/to/weights_dir" \
#   --loadWeights "eomt_cityscapes.bin" \
#   --temperature 1.0 \
#   --method msp \
#   --out "/path/out/best_fs_static.png" \
#   --exclude_no_ood

import os
import glob
import random
import sys
from argparse import ArgumentParser
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from mpl_toolkits.axes_grid1 import make_axes_locatable

from compute_metrics import get_metrics

# -----------------------------------------------------------------------------
# IMPORT EoMT (aggiungo la cartella eomt al PYTHONPATH)
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")
if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

from models.vit import ViT
from models.eomt import EoMT
from training.lightning_module import LightningModule

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 19
IMG_SIZE = (1024, 1024)  # (H,W)
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"
IGNORE_LABEL = 255

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([Resize(IMG_SIZE, Image.BILINEAR), ToTensor()])

# -----------------------------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------------------------
def load_eomt_model(ckpt_path: str, device: torch.device, masked_attn_enabled: bool = True) -> EoMT:
    encoder = ViT(img_size=IMG_SIZE, backbone_name=BACKBONE_NAME)
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=bool(masked_attn_enabled),
    )

    lm = LightningModule(
        network=network,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        attn_mask_annealing_enabled=False,
        attn_mask_annealing_start_steps=None,
        attn_mask_annealing_end_steps=None,
        lr=1e-4,
        llrd=0.8,
        llrd_l2_enabled=True,
        lr_mult=1.0,
        weight_decay=0.05,
        poly_power=0.9,
        warmup_steps=(500, 1000),
        ckpt_path=ckpt_path,
        delta_weights=False,
        load_ckpt_class_head=True,
    )

    model = lm.network
    model.to(device).eval()
    return model

# -----------------------------------------------------------------------------
# GT loader
# -----------------------------------------------------------------------------
def load_ood_gt_from_img_path(img_path: str, out_size=None) -> np.ndarray:
    pathGT = img_path.replace("images", "labels_masks")
    pathGT = os.path.splitext(pathGT)[0] + ".png"

    gt_pil = Image.open(pathGT).convert("L")
    if out_size is not None:
        gt_pil = gt_pil.resize((out_size[1], out_size[0]), resample=Image.NEAREST)  # (W,H)

    gt = np.array(gt_pil)
    uniq = set(np.unique(gt).tolist())

    if uniq.issubset({0, 2}):
        out = np.zeros_like(gt, dtype=np.uint8)
        out[gt == 2] = 1
        return out

    if uniq.issubset({0, 1, 255}) or uniq.issubset({0, 255}):
        return gt.astype(np.uint8, copy=False)

    raise ValueError(f"Unexpected GT values {sorted(uniq)} in {pathGT}")

# -----------------------------------------------------------------------------
# Pixel scores with temperature
# -----------------------------------------------------------------------------
@torch.no_grad()
def per_pixel_scores_with_temperature(mask_logits: torch.Tensor, class_logits: torch.Tensor, T: float) -> torch.Tensor:
    if T <= 0:
        raise ValueError("temperature must be > 0")
    mask_probs = mask_logits.sigmoid()
    class_probs = torch.softmax(class_logits / T, dim=-1)[..., :-1]  # drop void class
    return torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)

# -----------------------------------------------------------------------------
# Anomaly maps
# -----------------------------------------------------------------------------
@torch.no_grad()
def anomaly_map_from_pixel_scores(pixel_scores_chw: torch.Tensor, method: str, eps: float = 1e-8) -> torch.Tensor:
    method = method.lower()
    probs = torch.softmax(pixel_scores_chw, dim=0)

    if method == "msp":
        return 1.0 - probs.max(dim=0).values
    if method == "maxentropy":
        return -(probs * probs.clamp_min(eps).log()).sum(dim=0)
    if method == "maxlogit":
        return -pixel_scores_chw.max(dim=0).values

    raise ValueError(f"Unknown method: {method}")

# -----------------------------------------------------------------------------
# Save 2x2 figure
# -----------------------------------------------------------------------------
def save_figure_2x2(out_path: str, img_rgb: np.ndarray, gt_ood: np.ndarray, pred_sem: np.ndarray, anomaly: np.ndarray):
    """Save a clean 2x2 figure: Input / GT(OOD) / Prediction / Anomaly.

    Uses a dedicated colorbar axis (axes_grid1) to avoid squeezing other subplots.
    """
    gt_vis = gt_ood.copy()
    gt_vis[gt_vis == 255] = 0

    # Use constrained_layout but tighten its padding to reduce white space
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    try:
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    except Exception:
        pass

    # Ensure axes are centered and have equal aspect
    for ax in axes.ravel():
        ax.set_anchor("C")
        ax.set_aspect("equal")

    # --- Input ---
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Input", pad=4)
    axes[0, 0].axis("off")

    # --- GT ---
    axes[0, 1].imshow(gt_vis, vmin=0, vmax=1, interpolation="nearest")
    axes[0, 1].set_title("Ground Truth (OOD mask)", pad=4)
    axes[0, 1].axis("off")

    # --- Prediction ---
    axes[1, 0].imshow(pred_sem, interpolation="nearest")
    axes[1, 0].set_title("Prediction (semantic id)", pad=4)
    axes[1, 0].axis("off")

    # --- Anomaly ---
    im = axes[1, 1].imshow(anomaly, interpolation="nearest")
    axes[1, 1].set_title("Anomaly score", pad=4)
    axes[1, 1].axis("off")

    # Add a colorbar without messing up the grid geometry
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="3%", pad=0.02)
    fig.colorbar(im, cax=cax)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Glob, es: '/.../images/*.jpg'")
    parser.add_argument("--loadDir", required=True)
    parser.add_argument("--loadWeights", default="eomt_cityscapes.bin")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--method", type=str, default="msp", choices=["msp", "maxentropy", "maxlogit"])
    parser.add_argument("--out", type=str, required=True, help="Path output PNG")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--exclude_no_ood", action="store_true")
    parser.add_argument("--masked_attn_enabled", action="store_true", help="Se presente abilita masked_attn (default False)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt_path = os.path.join(args.loadDir, args.loadWeights)
    print("Loading:", ckpt_path)
    model = load_eomt_model(ckpt_path, device, masked_attn_enabled=args.masked_attn_enabled)
    print("Loaded on:", device)

    # build image list
    img_paths: List[str] = []
    for pat in args.input:
        img_paths.extend(glob.glob(os.path.expanduser(str(pat))))
    img_paths = sorted(list(set(img_paths)))

    if not img_paths:
        raise FileNotFoundError(f"No images matched input: {args.input}")
    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    best_path: Optional[str] = None
    best_ap: Optional[float] = None
    best_pack: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    for i, path in enumerate(img_paths, 1):
        try:
            img_pil = Image.open(path).convert("RGB")
            img_tensor = input_transform(img_pil).unsqueeze(0).float().to(device)  # [1,3,H,W]
            ood_gt = load_ood_gt_from_img_path(path, out_size=IMG_SIZE)  # [H,W]

            if args.exclude_no_ood and not (ood_gt == 1).any():
                continue

            with torch.no_grad():
                mask_logits_per_layer, class_logits_per_layer = model(img_tensor)
                mask_logits = F.interpolate(mask_logits_per_layer[-1], size=IMG_SIZE, mode="bilinear", align_corners=False)
                class_logits = class_logits_per_layer[-1]

                pixel_scores = per_pixel_scores_with_temperature(mask_logits, class_logits, args.temperature)[0]  # [C,H,W]
                pred_sem = torch.argmax(pixel_scores, dim=0).detach().cpu().numpy().astype(np.int32)

                amap = anomaly_map_from_pixel_scores(pixel_scores, args.method)
                amap_np = amap.detach().cpu().numpy().astype(np.float32, copy=False)

            valid = (ood_gt <= 1)
            flat_labels = ood_gt[valid].astype(np.uint8, copy=False).reshape(-1)
            flat_pred = amap_np[valid].astype(np.float32, copy=False).reshape(-1)

            if int(np.sum(flat_labels == 1)) == 0:
                continue

            res = get_metrics(flat_labels, flat_pred, num_points=50)
            ap = float(res["AP"])  # 0..1

            if (best_ap is None) or (ap > best_ap):
                best_ap = ap
                best_path = path
                img_rgb = np.array(img_pil.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)).astype(np.float32) / 255.0
                best_pack = (img_rgb, ood_gt, pred_sem, amap_np)

            if i % 50 == 0:
                print(f"[{i}/{len(img_paths)}] current best AP={0 if best_ap is None else best_ap:.4f}")

            del img_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Skip {path}: {e}")

    if best_pack is None or best_path is None:
        raise RuntimeError("No valid image found (check paths/GT and exclude_no_ood).")

    img_rgb, gt_ood, pred_sem, anomaly = best_pack
    save_figure_2x2(args.out, img_rgb, gt_ood, pred_sem, anomaly)

    print("\nBEST IMAGE:", best_path)
    print("BEST AP:", float(best_ap) * 100.0)
    print("SAVED:", args.out)

if __name__ == "__main__":
    main()