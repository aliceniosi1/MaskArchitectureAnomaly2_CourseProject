# evaluation_ts.py
# Copyright (c) OpenMMLab. All rights reserved.

import os
import glob
import random
import sys
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize, ToTensor

# -----------------------------------------------------------------------------
# IMPORT EoMT (aggiungiamo la cartella eomt al PYTHONPATH)
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
# CONFIG DI BASE (coerente con CityscapesSemantic / EoMT base_640)
# -----------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 19
IMG_SIZE = (1024, 1024)  # (H,W) per torchvision Resize va bene come tuple (H,W)
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize(IMG_SIZE, Image.BILINEAR),
        ToTensor(),  # -> float in [0,1]
    ]
)

target_transform = Compose(
    [
        Resize(IMG_SIZE, Image.NEAREST),
    ]
)

IGNORE_LABEL = 255


# -----------------------------------------------------------------------------
# METRICA FPR@95TPR
# -----------------------------------------------------------------------------
def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    scores: anomaly score (più alto => più OOD)
    labels: 1=OOD, 0=IND
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0
    return float(fpr[idxs[0]])


# -----------------------------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------------------------
def load_eomt_model(ckpt_path: str, device: torch.device) -> EoMT:
    encoder = ViT(
        img_size=IMG_SIZE,
        backbone_name=BACKBONE_NAME,
    )
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=True,
    )

    # LightningModule usato solo per caricare facilmente i pesi nel network
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
# GT loading + normalization (robusta)
# -----------------------------------------------------------------------------
def load_ood_gt_from_img_path(img_path: str) -> np.ndarray:
    """
    Restituisce una mappa HxW con valori:
      1 = OOD
      0 = IND
      255 = IGNORE (se presente)
    """
    pathGT = img_path.replace("images", "labels_masks")

    # estensioni tipiche
    lower = pathGT.lower()
    if "roadobsticle21" in lower or "roadobstacle21" in lower or "roadanomaly21" in lower:
        pathGT = os.path.splitext(pathGT)[0] + ".png"
    if "fs_static" in lower:
        pathGT = os.path.splitext(pathGT)[0] + ".png"
    if "roadanomaly" in lower:
        pathGT = os.path.splitext(pathGT)[0] + ".png"

    mask = Image.open(pathGT)
    mask = target_transform(mask)
    ood_gts = np.array(mask)

    # --- Regole storiche (come nei tuoi script) ---
    # RoadAnomaly (vecchio): spesso {0,2} dove 2=OOD
    uniq = np.unique(ood_gts)
    if 2 in uniq and 1 not in uniq:
        ood_gts = np.where(ood_gts == 2, 1, 0).astype(np.uint8)

    # LostAndFound (se usi quella versione legacy)
    if "lostandfound" in pathGT.lower():
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts).astype(np.uint8)

    # Streethazard legacy
    if "streethazard" in pathGT.lower():
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts).astype(np.uint8)

    return ood_gts


# -----------------------------------------------------------------------------
# Combine (mask_logits, class_logits) with temperature on class logits
# -----------------------------------------------------------------------------
@torch.no_grad()
def per_pixel_scores_with_temperature(
    mask_logits: torch.Tensor,   # [B,Q,h,w]
    class_logits: torch.Tensor,  # [B,Q,C+1]
    temperature: float,
) -> torch.Tensor:
    """
    Replica il concetto di to_per_pixel_logits_semantic, ma con Temperature Scaling:
      class_probs_T = softmax(class_logits / T)
      pixel_scores  = einsum(sigmoid(mask_logits), class_probs_T[...,:-1])
    Output: [B,C,H,W] (valori >=0, non normalizzati a 1)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    # masks -> [0,1]
    mask_probs = mask_logits.sigmoid()

    # class probs (drop no-object)
    class_probs = torch.softmax(class_logits / temperature, dim=-1)[..., :-1]  # [B,Q,C]

    # [B,Q,H,W] x [B,Q,C] -> [B,C,H,W]
    pixel_scores = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)
    return pixel_scores


# -----------------------------------------------------------------------------
# MSP-only anomaly map function
# -----------------------------------------------------------------------------
@torch.no_grad()
def anomaly_msp_from_pixel_scores(pixel_scores_chw: torch.Tensor) -> torch.Tensor:
    """\
    pixel_scores_chw: [C,H,W]
    returns: [H,W] MSP anomaly score (higher => more OOD)

    MSP anomaly score = 1 - max_c softmax(pixel_scores)_c
    NOTE: pixel_scores are non-negative class score maps (from mask-class combination).
    We apply softmax across classes to obtain a per-pixel distribution.
    """
    probs = torch.softmax(pixel_scores_chw, dim=0)
    return 1.0 - probs.max(dim=0).values


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True,
                        help="Glob, es: '/content/.../images/*.jpg'")
    parser.add_argument("--loadDir", required=True)
    parser.add_argument("--loadWeights", default="eomt_cityscapes.bin")
    parser.add_argument("--temperatures", nargs="+", type=float, required=True,
                        help="Lista di T, es: 0.5 0.75 1.0 1.5 2.0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1,
                        help="Per debug. -1 = tutte.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt_path = os.path.join(args.loadDir, args.loadWeights)
    print("Loading EoMT checkpoint:", ckpt_path)
    model = load_eomt_model(ckpt_path, device)
    print("Model LOADED successfully (EoMT + DINOv2)")

    # Build list immagini
    img_paths: List[str] = []
    for pat in args.input:
        img_paths.extend(glob.glob(os.path.expanduser(str(pat))))
    img_paths = sorted(list(set(img_paths)))

    if not img_paths:
        raise FileNotFoundError(f"No images matched input: {args.input}")

    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    temps = [float(t) for t in args.temperatures]

    # Accumulator (MSP only): scores[T] -> {"ood": [np arrays], "ind": [np arrays]}
    scores: Dict[float, Dict[str, List[np.ndarray]]] = {
        t: {"ood": [], "ind": []} for t in temps
    }

    processed = 0
    skipped_no_ood = 0
    errors = 0

    for idx, path in enumerate(img_paths, 1):
        try:
            # --- image ---
            img_pil = Image.open(path).convert("RGB")
            images = input_transform(img_pil).unsqueeze(0).float().to(device)  # [1,3,H,W]

            # --- gt ---
            ood_gts = load_ood_gt_from_img_path(path)  # HxW
            if 1 not in np.unique(ood_gts):
                skipped_no_ood += 1
                continue

            # Masks
            ood_mask = (ood_gts == 1)
            ind_mask = (ood_gts == 0)

            with torch.no_grad():
                mask_logits_per_layer, class_logits_per_layer = model(images)
                mask_logits = mask_logits_per_layer[-1]   # [B,Q,h,w]
                class_logits = class_logits_per_layer[-1] # [B,Q,C+1]

                # upscale masks to IMG_SIZE
                mask_logits = F.interpolate(
                    mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False
                )

                for T in temps:
                    pixel_scores_bchw = per_pixel_scores_with_temperature(
                        mask_logits=mask_logits,
                        class_logits=class_logits,
                        temperature=T,
                    )
                    pixel_scores = pixel_scores_bchw[0]  # [C,H,W]

                    amap = anomaly_msp_from_pixel_scores(pixel_scores)  # [H,W]
                    amap_np = amap.detach().cpu().numpy().astype(np.float32)

                    scores[T]["ood"].append(amap_np[ood_mask])
                    scores[T]["ind"].append(amap_np[ind_mask])

            processed += 1

            if idx % 10 == 0:
                print(f"[{idx}/{len(img_paths)}] processed={processed} skipped_no_ood={skipped_no_ood} errors={errors}")

            # free
            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            print(f"ERROR on {path}: {e}")

    print("\n--- DONE FORWARD ---")
    print(f"Images matched:      {len(img_paths)}")
    print(f"Processed (with OOD):{processed}")
    print(f"Skipped (no OOD):    {skipped_no_ood}")
    print(f"Errors:              {errors}")

    # -------------------------------------------------------------------------
    # Compute metrics and print tables
    # -------------------------------------------------------------------------
    print("\n=== RESULTS (MSP with Temperature Scaling) ===")
    print("Temp     |  AuPRC (%) | FPR@95 (%)")
    print("-----------------------------------")

    best_T = None
    best_auprc = -1.0

    for T in temps:
        ood_list = scores[T]["ood"]
        ind_list = scores[T]["ind"]

        if len(ood_list) == 0 or len(ind_list) == 0:
            print(f"{T:<8.3f} |    (no data) |   (no data)")
            continue

        ood_out = np.concatenate(ood_list, axis=0)
        ind_out = np.concatenate(ind_list, axis=0)

        val_out = np.concatenate((ind_out, ood_out), axis=0)
        val_label = np.concatenate(
            (np.zeros_like(ind_out, dtype=np.uint8), np.ones_like(ood_out, dtype=np.uint8)),
            axis=0
        )

        prc_auc = average_precision_score(val_label, val_out) * 100.0
        fpr95 = fpr_at_95_tpr(val_out, val_label) * 100.0

        if prc_auc > best_auprc:
            best_auprc = prc_auc
            best_T = T

        print(f"{T:<8.3f} | {prc_auc:10.2f} | {fpr95:9.2f}")

    if best_T is not None:
        print(f"\nBest T by AuPRC: T={best_T} (AuPRC={best_auprc:.2f}%)")


if __name__ == "__main__":
    main()