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
from torchvision.transforms import Compose, Resize, ToTensor
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
# CONFIG DI BASE
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
        ToTensor(),
    ]
)

IGNORE_LABEL = 255

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
# GT loader: tiene {0,1,255} invariato, converte RoadAnomaly {0,2} -> {0,1}
# -----------------------------------------------------------------------------
def load_ood_gt_from_img_path(img_path: str, out_size=None) -> np.ndarray:
    pathGT = img_path.replace("images", "labels_masks")
    pathGT = os.path.splitext(pathGT)[0] + ".png"

    gt_pil = Image.open(pathGT).convert("L")

    # Se ridimensioni le immagini in input, ridimensiona anche la GT per allinearla
    if out_size is not None:
        gt_pil = gt_pil.resize((out_size[1], out_size[0]), resample=Image.NEAREST)  # (W,H)

    gt = np.array(gt_pil)
    uniq = set(np.unique(gt).tolist())

    # RoadAnomaly: {0,2} con 2=OOD
    if uniq.issubset({0, 2}):
        out = np.zeros_like(gt, dtype=np.uint8)
        out[gt == 2] = 1
        return out

    # Standard: {0,1,255}
    if uniq.issubset({0, 1, 255}):
        return gt.astype(np.uint8, copy=False)

    # Alcuni casi possono essere {0,255} (immagini senza OOD)
    if uniq.issubset({0, 255}):
        return gt.astype(np.uint8, copy=False)

    raise ValueError(f"Unexpected GT values {sorted(uniq)} in {pathGT}")

# -----------------------------------------------------------------------------
# Combine (mask_logits, class_logits) with temperature on class logits
# -----------------------------------------------------------------------------
@torch.no_grad()
def per_pixel_scores_with_temperature(
    mask_logits: torch.Tensor,   # [B,Q,H,W]
    class_logits: torch.Tensor,  # [B,Q,C+1]
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    mask_probs = mask_logits.sigmoid()
    class_probs = torch.softmax(class_logits / temperature, dim=-1)[..., :-1]  # drop void class

    pixel_scores = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)
    return pixel_scores

# -----------------------------------------------------------------------------
# OOD scoring methods from pixel scores
# -----------------------------------------------------------------------------
@torch.no_grad()
def anomaly_map_from_pixel_scores(
    pixel_scores_chw: torch.Tensor, method: str, eps: float = 1e-8
) -> torch.Tensor:
    method = method.lower()
    probs = torch.softmax(pixel_scores_chw, dim=0)

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
    parser.add_argument("--max_images", type=int, default=-1, help="Debug. -1=tutte.")
    parser.add_argument(
        "--out_csv", type=str, default=None,
        help="Path CSV output (opzionale). Se non specificato, non salva nulla."
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

    # Accumulator per get_metrics: per (T,method) salvo labels/pred flatten (solo valid 0/1)
    acc: Dict[Tuple[float, str], Dict[str, List[np.ndarray]]] = {
        (float(t), m): {"labels": [], "pred": []} for t in temps for m in methods
    }

    processed = 0
    skipped_no_ood = 0
    errors = 0

    for idx, path in enumerate(img_paths, 1):
        try:
            img_pil = Image.open(path).convert("RGB")
            images = input_transform(img_pil).unsqueeze(0).float().to(device)  # [1,3,H,W]

            ood_gts = load_ood_gt_from_img_path(path, out_size=IMG_SIZE)  # HxW


            with torch.no_grad():
                mask_logits_per_layer, class_logits_per_layer = model(images)
                mask_logits = mask_logits_per_layer[-1]    # [B,Q,h,w]
                class_logits = class_logits_per_layer[-1]  # [B,Q,C+1]

                # upsample masks -> IMG_SIZE
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
                        amap_np = amap.detach().cpu().numpy().astype(np.float32, copy=False)

                        valid_mask = (ood_gts <= 1)  # 0/1, escludo 255
                        flat_labels = ood_gts[valid_mask].astype(np.uint8, copy=False).reshape(-1)
                        flat_pred = amap_np[valid_mask].astype(np.float32, copy=False).reshape(-1)

                        key = (float(T), method)
                        if flat_labels.size > 0:
                            acc[key]["labels"].append(flat_labels)
                            acc[key]["pred"].append(flat_pred)

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
    # Metrics via get_metrics (che hai già)
    # -------------------------------------------------------------------------
    print("\n=== RESULTS (Fishyscapes get_metrics) ===")
    print("Method      Temp     |  AUPRC(%)   AUROC(%)  FPR@95(%)")
    print("-------------------------------------------------------")

    best_by_method: Dict[str, Tuple[float, float]] = {}
    rows: List[dict] = []

    for (T, method) in sorted(acc.keys(), key=lambda x: (x[1], x[0])):
        lab_list = acc[(T, method)]["labels"]
        pred_list = acc[(T, method)]["pred"]

        if len(lab_list) == 0:
            print(f"{method:<10s} {T:<8.3f} | (no valid pixels)")
            continue

        flat_labels = np.concatenate(lab_list, axis=0)
        flat_pred = np.concatenate(pred_list, axis=0)

        # Se non ci sono positivi in tutto il set, AP/FPR@95 non sono definiti
        n_pos = int(np.sum(flat_labels == 1))
        if n_pos == 0:
            print(f"{method:<10s} {T:<8.3f} | (no positives in GT)")
            continue

        # get_metrics deve essere definita/importata da te (tu hai già la funzione)
        res = get_metrics(flat_labels, flat_pred, num_points=50)

        ap = float(res["AP"]) * 100.0
        auroc = float(res["auroc"]) * 100.0
        fpr95 = float(res["FPR@95%TPR"]) * 100.0

        print(f"{method:<10s} {T:<8.3f} | {ap:7.2f}  {auroc:8.2f}  {fpr95:9.2f}")

        rows.append({
            "method": str(method),
            "temperature": float(T),
            "AP_percent": float(ap),
            "AUROC_percent": float(auroc),
            "FPR95_percent": float(fpr95),
            "n_pixels_valid": int(flat_labels.size),
            "n_pos_pixels": int(n_pos),
            "images_matched": int(len(img_paths)),
            "images_processed": int(processed),
            "images_without_ood": int(skipped_no_ood),
            "errors": int(errors),
            "weights": str(ckpt_path),
            "img_size_h": int(IMG_SIZE[0]),
            "img_size_w": int(IMG_SIZE[1]),
        })

        if (method not in best_by_method) or (ap > best_by_method[method][1]):
            best_by_method[method] = (float(T), float(ap))

    for m, (bt, ba) in best_by_method.items():
        print(f"\nBest T by AUPRC for {m}: T={bt} (AP={ba:.2f}%)")

    # ---------------------------------------------------------------------
    # Save CSV (optional)
    # ---------------------------------------------------------------------
    if args.out_csv is not None:
        out_csv = args.out_csv
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # aggiungo info best per method in ogni riga
        for r in rows:
            m = r["method"]
            best = best_by_method.get(m, (None, None))
            r["best_T_by_AP_for_method"] = best[0]
            r["best_AP_percent_for_method"] = best[1]

        fieldnames = [
            "method", "temperature",
            "AP_percent", "AUROC_percent", "FPR95_percent",
            "n_pixels_valid", "n_pos_pixels",
            "images_matched", "images_processed", "images_without_ood", "errors",
            "weights", "img_size_h", "img_size_w",
            "best_T_by_AP_for_method", "best_AP_percent_for_method",
        ]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nCSV salvato in: {out_csv}")

if __name__ == "__main__":
    main()