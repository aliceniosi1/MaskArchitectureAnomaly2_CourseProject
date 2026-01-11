# eval_semantic_ts.py

import os
import sys
import glob
import csv
import random
from contextlib import nullcontext
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from PIL import Image
from torchvision.transforms import Compose, Resize, PILToTensor

from compute_metrics import get_metrics

# -----------------------------------------------------------------------------
# Import EoMT (aggiungo eomt al PYTHONPATH)
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
# Config base
# -----------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 19
IMG_SIZE = (1024, 1024)  # (H, W)
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

IGNORE_LABEL = 255

# NOTA: uso PILToTensor -> uint8 [0..255], più robusto con window_imgs_semantic
input_transform = Compose(
    [
        Resize(IMG_SIZE, Image.BILINEAR),
        PILToTensor(),  # uint8, shape [C,H,W]
    ]
)

# -----------------------------------------------------------------------------
# Loader modello (ritorna LightningModule, perché infer_semantic usa windowing)
# -----------------------------------------------------------------------------
def load_eomt_lightning(ckpt_path: str, device: torch.device, masked_attn_enabled: bool = False) -> LightningModule:
    encoder = ViT(img_size=IMG_SIZE, backbone_name=BACKBONE_NAME)
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=masked_attn_enabled,
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

    lm.to(device).eval()
    return lm

# -----------------------------------------------------------------------------
# GT loader (come hai scritto tu)
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

    if uniq.issubset({0, 1, 255}):
        return gt.astype(np.uint8, copy=False)

    if uniq.issubset({0, 255}):
        return gt.astype(np.uint8, copy=False)

    raise ValueError(f"Unexpected GT values {sorted(uniq)} in {pathGT}")

# -----------------------------------------------------------------------------
# Infer semantic (versione “notebook style” + temperatura)
# target qui NON serve per l’OOD (lo lasciamo solo per compatibilità)
# -----------------------------------------------------------------------------
def infer_semantic(img: torch.Tensor, target, T: float, model: LightningModule, device: torch.device) -> torch.Tensor:
    ctx = autocast(dtype=torch.float16, device_type="cuda") if device.type == "cuda" else nullcontext()

    with torch.no_grad(), ctx:
        # img: [3,H,W]
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = model.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], IMG_SIZE, mode="bilinear", align_corners=False
        )

        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1] / float(T)
        )
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

    return logits  # [B,C,H,W]

# -----------------------------------------------------------------------------
# OOD score methods from per-pixel logits/scores
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
# Main
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help='Glob, es: "/.../images/*.jpg"')
    parser.add_argument("--loadDir", required=True)
    parser.add_argument("--loadWeights", default="eomt_cityscapes.bin")
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--methods", nargs="+", default=["msp"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--exclude_no_ood", action="store_true",
                        help="Se attivo, scarta immagini senza nessun pixel OOD (=1).")
    parser.add_argument("--masked_attn_enabled", action="store_true",
                    help="Se presente abilita masked attention. Default: False.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt_path = os.path.join(args.loadDir, args.loadWeights)
    print("Loading EoMT checkpoint:", ckpt_path)
    model = load_eomt_lightning(ckpt_path, device, masked_attn_enabled=args.masked_attn_enabled)
    print("Model LOADED successfully")

    # parse methods (supporta virgole)
    methods: List[str] = []
    for m in args.methods:
        methods.extend([x.strip() for x in str(m).split(",") if x.strip()])
    methods = [m.lower() for m in methods]

    temps = [float(t) for t in args.temperatures]

    # build image list
    img_paths: List[str] = []
    for pat in args.input:
        img_paths.extend(glob.glob(os.path.expanduser(str(pat))))
    img_paths = sorted(list(set(img_paths)))

    if not img_paths:
        raise FileNotFoundError(f"No images matched input: {args.input}")
    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    # accumulo labels/pred per get_metrics
    acc: Dict[Tuple[float, str], Dict[str, List[np.ndarray]]] = {
        (float(t), m): {"labels": [], "pred": []} for t in temps for m in methods
    }

    processed = 0
    skipped_no_ood = 0
    errors = 0

    for idx, path in enumerate(img_paths, 1):
        try:
            img_pil = Image.open(path).convert("RGB")
            img = input_transform(img_pil)  # uint8 [3,H,W]
            img = img.to(device)

            ood_gts = load_ood_gt_from_img_path(path, out_size=IMG_SIZE)  # [H,W]

            has_ood = bool((ood_gts == 1).any())
            if args.exclude_no_ood and (not has_ood):
                skipped_no_ood += 1
                continue

            valid_mask = (ood_gts <= 1)  # include 0/1, esclude 255
            flat_labels_all = ood_gts[valid_mask].astype(np.uint8, copy=False).reshape(-1)

            for T in temps:
                logits_bchw = infer_semantic(img=img, target=None, T=T, model=model, device=device)
                pixel_scores = logits_bchw[0]  # [C,H,W]

                # se c'è una classe extra (void), la droppo
                if pixel_scores.shape[0] == NUM_CLASSES + 1:
                    pixel_scores = pixel_scores[:-1]

                for method in methods:
                    amap = anomaly_map_from_pixel_scores(pixel_scores, method=method)  # [H,W]
                    amap_np = amap.detach().cpu().numpy().astype(np.float32, copy=False)
                    flat_pred_all = amap_np[valid_mask].astype(np.float32, copy=False).reshape(-1)

                    key = (float(T), method)
                    if flat_labels_all.size > 0:
                        acc[key]["labels"].append(flat_labels_all)
                        acc[key]["pred"].append(flat_pred_all)

            processed += 1
            if idx % 10 == 0:
                print(f"[{idx}/{len(img_paths)}] processed={processed} skipped_no_ood={skipped_no_ood} errors={errors}")

            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            print(f"ERROR on {path}: {e}")

    print("\n--- DONE FORWARD ---")
    print(f"Images matched:      {len(img_paths)}")
    print(f"Processed:           {processed}")
    print(f"Images skipped(no OOD): {skipped_no_ood}" if args.exclude_no_ood else f"Images without OOD (not excluded): {sum(1 for p in img_paths if not (load_ood_gt_from_img_path(p, out_size=IMG_SIZE)==1).any())}")
    print(f"Errors:              {errors}")

    # metrics
    print("\n=== RESULTS (get_metrics) ===")
    print("Method      Temp     |  AP(%)     AUROC(%)   FPR@95(%)")
    print("-------------------------------------------------------")

    rows: List[dict] = []
    best_by_method: Dict[str, Tuple[float, float]] = {}

    for (T, method) in sorted(acc.keys(), key=lambda x: (x[1], x[0])):
        lab_list = acc[(T, method)]["labels"]
        pred_list = acc[(T, method)]["pred"]
        if len(lab_list) == 0:
            print(f"{method:<10s} {T:<8.3f} | (no valid pixels)")
            continue

        flat_labels = np.concatenate(lab_list, axis=0)
        flat_pred = np.concatenate(pred_list, axis=0)

        n_pos = int(np.sum(flat_labels == 1))
        if n_pos == 0:
            print(f"{method:<10s} {T:<8.3f} | (no positives in GT)")
            continue

        res = get_metrics(flat_labels, flat_pred, num_points=50)
        ap = float(res["AP"]) * 100.0
        auroc = float(res["auroc"]) * 100.0
        fpr95 = float(res["FPR@95%TPR"]) * 100.0

        print(f"{method:<10s} {T:<8.3f} | {ap:7.2f}   {auroc:8.2f}   {fpr95:9.2f}")

        rows.append({
            "method": method,
            "temperature": float(T),
            "AP_percent": float(ap),
            "AUROC_percent": float(auroc),
            "FPR95_percent": float(fpr95),
            "n_pixels_valid": int(flat_labels.size),
            "n_pos_pixels": int(n_pos),
            "images_matched": int(len(img_paths)),
            "images_processed": int(processed),
            "excluded_no_ood": bool(args.exclude_no_ood),
            "errors": int(errors),
            "weights": str(ckpt_path),
            "img_size_h": int(IMG_SIZE[0]),
            "img_size_w": int(IMG_SIZE[1]),
        })

        if (method not in best_by_method) or (ap > best_by_method[method][1]):
            best_by_method[method] = (float(T), float(ap))

    for m, (bt, ba) in best_by_method.items():
        print(f"\nBest T by AP for {m}: T={bt} (AP={ba:.2f}%)")

    # save csv
    if args.out_csv is not None:
        out_csv = args.out_csv
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        for r in rows:
            m = r["method"]
            bestT, bestAP = best_by_method.get(m, (None, None))
            r["best_T_by_AP_for_method"] = bestT
            r["best_AP_percent_for_method"] = bestAP

        fieldnames = list(rows[0].keys()) if rows else [
            "method","temperature","AP_percent","AUROC_percent","FPR95_percent",
            "n_pixels_valid","n_pos_pixels","images_matched","images_processed",
            "excluded_no_ood","errors","weights","img_size_h","img_size_w",
            "best_T_by_AP_for_method","best_AP_percent_for_method"
        ]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nCSV salvato in: {out_csv}")

if __name__ == "__main__":
    main()