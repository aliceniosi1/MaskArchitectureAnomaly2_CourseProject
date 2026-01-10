# evaluation_ts.py
# Copyright (c) OpenMMLab. All rights reserved.

import os
import glob
import random
import sys
import csv
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize, ToTensor

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
# CONFIG DI BASE (coerente con CityscapesSemantic / EoMT base_640, adattata al tuo eval)
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
# METRICA FPR@95TPR (prendo il minimo FPR tra i punti con TPR>=0.95)
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
    return float(np.min(fpr[idxs]))


# -----------------------------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------------------------
def load_eomt_model(ckpt_path: str, device: torch.device) -> EoMT:
    """
    Nota: in questo progetto spesso ckpt_path può essere sia .ckpt sia .bin
    (perché LightningModule gestisce ckpt_path internamente).
    """
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
# GT loading + normalization (robusta) -> output sempre {0,1,255}
# -----------------------------------------------------------------------------
def load_ood_gt_from_img_path(img_path: str) -> np.ndarray:
    """
    Ritorna HxW con valori:
      0   = IND
      1   = OOD
      255 = IGNORE
    """
    pathGT = img_path.replace("images", "labels_masks")

    # fix estensioni
    lower = pathGT.lower()
    if "roadobsticle21" in lower or "roadobstacle21" in lower or "roadanomaly21" in lower:
        pathGT = os.path.splitext(pathGT)[0] + ".png"
    if "fs_static" in lower or "fishyscapes" in lower:
        pathGT = os.path.splitext(pathGT)[0] + ".png"
    if "roadanomaly" in lower:
        pathGT = os.path.splitext(pathGT)[0] + ".png"

    mask = Image.open(pathGT)
    mask = target_transform(mask)
    ood_gts = np.array(mask)

    uniq = set(np.unique(ood_gts).tolist())

    # RoadAnomaly legacy: {0,2} con 2=OOD
    if ("roadanomaly" in lower) and ("roadanomaly21" not in lower) and (2 in uniq) and (1 not in uniq):
        out = np.full_like(ood_gts, 255, dtype=np.uint8)
        out[ood_gts == 0] = 0
        out[ood_gts == 2] = 1
        return out

    # LostAndFound legacy (se capita ancora quel formato)
    if "lostandfound" in lower:
        # se già è {0,1,255} lo tengo
        if uniq.issubset({0, 1, 255}):
            return ood_gts.astype(np.uint8, copy=False)

        # altrimenti applico la logica legacy e ripulisco in {0,1,255}
        tmp = ood_gts.copy()
        tmp = np.where((tmp == 0), 255, tmp)
        tmp = np.where((tmp == 1), 0, tmp)
        tmp = np.where((tmp > 1) & (tmp < 201), 1, tmp).astype(np.uint8)

        out = np.full_like(tmp, 255, dtype=np.uint8)
        out[tmp == 0] = 0
        out[tmp == 1] = 1
        return out

    # Default: mi aspetto {0,1,255}
    if not uniq.issubset({0, 1, 255}):
        raise ValueError(f"Unexpected GT values {sorted(uniq)} in {pathGT} (expected subset of {{0,1,255}})")

    return ood_gts.astype(np.uint8, copy=False)


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
    Temperature scaling SOLO sui class logits:
      class_probs_T = softmax(class_logits / T)
      pixel_scores  = einsum(sigmoid(mask_logits), class_probs_T[...,:-1])

    Output: [B,C,H,W] (>=0, non normalizzato)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    mask_probs = mask_logits.sigmoid()  # [B,Q,H,W] in [0,1]
    class_probs = torch.softmax(class_logits / temperature, dim=-1)[..., :-1]  # [B,Q,C]

    pixel_scores = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)
    return pixel_scores


# -----------------------------------------------------------------------------
# OOD scoring methods from pixel scores
# -----------------------------------------------------------------------------
@torch.no_grad()
def anomaly_map_from_pixel_scores(pixel_scores_chw: torch.Tensor, method: str, eps: float = 1e-8) -> torch.Tensor:
    """
    pixel_scores_chw: [C,H,W] (>=0)
    method:
      - msp        : 1 - max softmax(pixel_scores)
      - maxentropy : entropy(softmax(pixel_scores))
      - maxlogit   : -max(pixel_scores)
    """
    method = method.lower()

    probs = torch.softmax(pixel_scores_chw, dim=0)  # [C,H,W]

    if method == "msp":
        return 1.0 - probs.max(dim=0).values

    if method == "maxentropy":
        ent = -(probs * probs.clamp_min(eps).log()).sum(dim=0)
        return ent

    if method == "maxlogit":
        return -pixel_scores_chw.max(dim=0).values

    raise ValueError(f"Unknown method: {method}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Glob, es: '/content/.../images/*.jpg'"
    )
    parser.add_argument("--loadDir", required=True)
    parser.add_argument("--loadWeights", default="eomt_cityscapes.bin")
    parser.add_argument(
        "--temperatures", nargs="+", type=float, required=True,
        help="Lista di T, es: 0.5 0.75 1.0 1.5 2.0"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["msp"],
        help="Metodi OOD: msp maxentropy maxlogit. Accetto anche virgole: --methods msp,maxentropy"
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1, help="Per debug. -1 = tutte.")
    parser.add_argument(
        "--out_csv", type=str, default=None,
        help="Path CSV output. Se non specificato, non salva nulla."
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt_path = os.path.join(args.loadDir, args.loadWeights)
    print("Loading EoMT checkpoint:", ckpt_path)
    model = load_eomt_model(ckpt_path, device)
    print("Model LOADED successfully (EoMT + DINOv2)")

    # Parse methods (accetto sia spazi che virgole)
    methods: List[str] = []
    for m in args.methods:
        methods.extend([x.strip() for x in str(m).split(",") if x.strip()])
    methods = [m.lower() for m in methods]

    temps = [float(t) for t in args.temperatures]

    print("Device:", device)
    print("IMG_SIZE:", IMG_SIZE)
    print("Temps:", temps)
    print("Methods:", methods)

    # Build list immagini
    img_paths: List[str] = []
    for pat in args.input:
        img_paths.extend(glob.glob(os.path.expanduser(str(pat))))
    img_paths = sorted(list(set(img_paths)))

    if not img_paths:
        raise FileNotFoundError(f"No images matched input: {args.input}")

    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    # Accumulator: scores[(T,method)] -> {"ood": [...], "ind": [...]}
    scores: Dict[Tuple[float, str], Dict[str, List[np.ndarray]]] = {
        (float(t), m): {"ood": [], "ind": []} for t in temps for m in methods
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
            ood_gts = load_ood_gt_from_img_path(path)  # HxW in {0,1,255}

            # Masks
            ood_mask = (ood_gts == 1)
            ind_mask = (ood_gts == 0)

            if not ood_mask.any():
                skipped_no_ood += 1

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

                    for method in methods:
                        amap = anomaly_map_from_pixel_scores(pixel_scores, method=method)  # [H,W]
                        amap_np = amap.detach().cpu().numpy().astype(np.float32)

                        ood_vals = amap_np[ood_mask]
                        ind_vals = amap_np[ind_mask]

                        key = (float(T), method)
                        if ood_vals.size > 0:
                            scores[key]["ood"].append(ood_vals)
                        if ind_vals.size > 0:
                            scores[key]["ind"].append(ind_vals)

            processed += 1

            if idx % 10 == 0:
                print(f"[{idx}/{len(img_paths)}] processed={processed} skipped_no_ood={skipped_no_ood} errors={errors}")

            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            print(f"ERROR on {path}: {e}")

    print("\n--- DONE FORWARD ---")
    print(f"Images matched:      {len(img_paths)}")
    print(f"Processed:           {processed}")
    print(f"Images without OOD:  {skipped_no_ood}")
    print(f"Errors:              {errors}")

    # -------------------------------------------------------------------------
    # Compute metrics and print tables
    # -------------------------------------------------------------------------
    print("\n=== RESULTS (Temperature Scaling) ===")
    print("Method      Temp     |  AuPRC (%) | FPR@95 (%)")
    print("------------------------------------------------")

    best_by_method: Dict[str, Tuple[float, float]] = {}  # method -> (best_T, best_auprc)
    rows: List[dict] = []

    for (T, method) in sorted(scores.keys(), key=lambda x: (x[1], x[0])):
        ood_list = scores[(T, method)]["ood"]
        ind_list = scores[(T, method)]["ind"]

        if len(ind_list) == 0 or len(ood_list) == 0:
            print(f"{method:<10s} {T:<8.3f} |   (no data) |   (no data)")
            continue

        ind_out = np.concatenate(ind_list, axis=0)
        ood_out = np.concatenate(ood_list, axis=0)

        if ind_out.size == 0 or ood_out.size == 0:
            print(f"{method:<10s} {T:<8.3f} |   (no data) |   (no data)")
            continue

        val_out = np.concatenate((ind_out, ood_out), axis=0)
        val_label = np.concatenate(
            (np.zeros_like(ind_out, dtype=np.uint8), np.ones_like(ood_out, dtype=np.uint8)),
            axis=0
        )

        prc_auc = average_precision_score(val_label, val_out) * 100.0
        fpr95 = fpr_at_95_tpr(val_out, val_label) * 100.0

        # update best per method
        if (method not in best_by_method) or (prc_auc > best_by_method[method][1]):
            best_by_method[method] = (float(T), float(prc_auc))

        print(f"{method:<10s} {T:<8.3f} | {prc_auc:10.2f} | {fpr95:9.2f}")

        rows.append({
            "method": str(method),
            "temperature": float(T),
            "auprc_percent": float(prc_auc),
            "fpr95_percent": float(fpr95),
            "n_ind_pixels": int(ind_out.size),
            "n_ood_pixels": int(ood_out.size),
            "images_matched": int(len(img_paths)),
            "images_processed": int(processed),
            "images_without_ood": int(skipped_no_ood),
            "errors": int(errors),
            "weights": str(ckpt_path),
            "img_size_h": int(IMG_SIZE[0]),
            "img_size_w": int(IMG_SIZE[1]),
        })

    for m, (bt, ba) in best_by_method.items():
        print(f"\nBest T by AuPRC for {m}: T={bt} (AuPRC={ba:.2f}%)")

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    if args.out_csv is not None:
        out_csv = args.out_csv
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # aggiungo best_T per method in ogni riga (comodo per leggere il CSV)
        for r in rows:
            m = r["method"]
            r["best_T_by_auprc_for_method"] = best_by_method.get(m, ("", ""))[0]
            r["best_auprc_for_method"] = best_by_method.get(m, ("", ""))[1]

        fieldnames = [
            "method", "temperature", "auprc_percent", "fpr95_percent",
            "n_ind_pixels", "n_ood_pixels",
            "images_matched", "images_processed", "images_without_ood", "errors",
            "weights", "img_size_h", "img_size_w",
            "best_T_by_auprc_for_method", "best_auprc_for_method",
        ]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nCSV salvato in: {out_csv}")


if __name__ == "__main__":
    main()