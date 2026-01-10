# eval_ood.py
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

from training.lightning_module import LightningModule


# -------------------------
# GT mapping (based on YOUR inspect)
# -------------------------
def map_gt_to_ood_binary(ood_gts: np.ndarray, dataset: str) -> np.ndarray:
    """
    Output mask:
      0   = in-distribution
      1   = OOD/anomaly
      255 = ignore/void

    YOUR inspect:
      - RoadAnomaly: {0,2}        -> 2 is OOD
      - RoadAnomaly21: {0,1,255}  -> 1 is OOD, 255 ignore
      - RoadObsticle21: {0,1,255}
      - LostAndFound: {0,1,255}
      - fs_static: {0,1,255}
    """
    ds = dataset.lower()
    u = set(np.unique(ood_gts).tolist())

    def as_uint8(x):
        return x.astype(np.uint8, copy=False)

    # RoadAnomaly: {0,2}
    if "roadanomaly" in ds and "roadanomaly21" not in ds:
        out = np.full_like(ood_gts, 255, dtype=np.uint8)  # default ignore
        out[ood_gts == 0] = 0
        out[ood_gts == 2] = 1
        # if any weird values appear -> stay 255
        return out

    # Others: already {0,1,255}
    if (
        "roadanomaly21" in ds
        or "roadobsticle21" in ds or "roadobstacle21" in ds
        or "lostandfound" in ds or "lost&found" in ds or "lost_and_found" in ds
        or "fs_static" in ds or "fishyscapes" in ds
    ):
        if u.issubset({0, 1, 255}):
            return as_uint8(ood_gts)
        raise ValueError(f"[{dataset}] Unexpected GT values {sorted(u)} (expected subset of {{0,1,255}})")

    raise ValueError(f"Unknown dataset: {dataset}")


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
    # your convention
    pathGT = path_img.replace("images", "labels_masks")
    p = pathGT.lower()

    # extension fixes (keep them)
    if ("roadobsticle21" in p or "roadobstacle21" in p) and pathGT.lower().endswith(".webp"):
        pathGT = pathGT[:-5] + ".png"
    if ("fs_static" in p or "fishyscapes" in p) and pathGT.lower().endswith(".jpg"):
        pathGT = pathGT[:-4] + ".png"
    if "roadanomaly" in p and pathGT.lower().endswith(".jpg"):
        pathGT = pathGT[:-4] + ".png"

    return pathGT


# -------------------------
# Metrics utils
# -------------------------
def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0
    return float(np.min(fpr[idxs]))

def parse_list_floats(s: str):
    s = s.replace(",", " ").strip()
    return [float(x) for x in s.split() if x.strip()]


def parse_list_str(s: str):
    s = s.replace(",", " ").strip()
    return [x.strip() for x in s.split() if x.strip()]


def anomaly_map_from_outputs(
    scores_chw: torch.Tensor,
    probs_chw: torch.Tensor,
    method: str,
    eps: float = 1e-8) -> torch.Tensor:
    """
    scores_chw: [C,H,W] per-pixel class scores (output di to_per_pixel_logits_semantic)
    probs_chw : [C,H,W] probs per pixel (scores normalizzati su C)
    """
    method = method.lower()

    if method == "msp":
        msp = probs_chw.max(dim=0).values
        return 1.0 - msp

    if method == "maxentropy":
        entropy = -(probs_chw * probs_chw.clamp_min(eps).log()).sum(dim=0)
        return entropy

    if method == "maxlogit":
        # proxy "max logit" nel tuo setup = -max score (più corretto di -max prob)
        maxs = scores_chw.max(dim=0).values
        return -maxs

    raise ValueError(f"Unsupported method: {method}")


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint .ckpt (from ModelCheckpoint)")
    parser.add_argument("--input", type=str, required=True, help='Glob images, e.g. "/path/RoadAnomaly/images/*.jpg"')
    parser.add_argument("--temps", type=str, default="1.0", help='Temperatures, e.g. "0.5,1,2,4"')
    parser.add_argument("--methods", type=str, default="msp", help='Methods, e.g. "msp,maxentropy,maxlogit"')
    parser.add_argument("--img_size", type=int, nargs=2, default=(640, 640), help="H W resize for eval")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    H, W = int(args.img_size[0]), int(args.img_size[1])
    IMG_SIZE = (H, W)

    temps = parse_list_floats(args.temps)
    methods = [m.lower() for m in parse_list_str(args.methods)]
    print("IMG_SIZE:", IMG_SIZE)
    print("Temps:", temps)
    print("Methods:", methods)

    # --- Load LightningModule from checkpoint
    # This restores hyperparameters saved by Lightning, and loads weights.
    lm: LightningModule = LightningModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    lm.to(device)
    lm.eval()

    # Ensure module uses the eval resize if you want
    lm.img_size = IMG_SIZE

    file_list = sorted(glob.glob(os.path.expanduser(args.input)))
    print(f"Found {len(file_list)} images")

    scores = defaultdict(list)  # key=(T,method)
    gts = defaultdict(list)

    for path in file_list:
        # image -> 0..255 float tensor (because lm.forward divides by 255)
        img_pil = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
        img_np = np.array(img_pil, dtype=np.uint8)  # H,W,3
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

        mask_logits_layers, class_logits_layers = lm(img_tensor)
        mask_logits = mask_logits_layers[-1]    # [B,Q,h,w]
        class_logits = class_logits_layers[-1]  # [B,Q,C+1]

        # upsample masks
        mask_logits = F.interpolate(mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False)

        # --- GT
        pathGT = default_gt_path_from_img(path)
        if not os.path.exists(pathGT):
            continue

        gt_img = Image.open(pathGT).resize((W, H), Image.NEAREST)
        ood_raw = np.array(gt_img)
        ds = infer_dataset_from_path(pathGT)
        ood_bin = map_gt_to_ood_binary(ood_raw, ds)

        if not np.any(ood_bin != 255):
            continue

        for T in temps:
            clsT = class_logits / float(T)

            # per-pixel class scores [B,C,H,W]
            per_pixel_scores = LightningModule.to_per_pixel_logits_semantic(mask_logits, clsT)
            scores_chw = per_pixel_scores[0]  # [C,H,W]
            # normalize to probs per pixel across classes
            probs = per_pixel_scores / per_pixel_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
            probs_chw = probs[0]  # [C,H,W]

            for method in methods:
                amap = anomaly_map_from_outputs(scores_chw, probs_chw, method)
                key = (float(T), method)
                scores[key].append(amap.detach().cpu().numpy())
                gts[key].append(ood_bin)

    if not scores:
        print("No valid samples collected.")
        return

    print("\n================ RESULTS ================")
    for key in sorted(scores.keys(), key=lambda x: (x[0], x[1])):
        T, method = key
        anomaly_scores = np.stack(scores[key], axis=0)
        ood_masks = np.stack(gts[key], axis=0)

        valid = (ood_masks != 255)
        labels = ood_masks[valid].astype(np.uint8)
        sc = anomaly_scores[valid].astype(np.float32)

        if labels.max() == labels.min():
            print(f"T={T:g} method={method:10s} -> skipped (only one class present)")
            continue

        auprc = average_precision_score(labels, sc)
        fpr95 = fpr_at_95_tpr(sc, labels)
        print(f"T={T:g}  method={method:10s} | AUPRC={auprc*100:6.2f}% | FPR@95TPR={fpr95*100:6.2f}%")

    print("========================================\n")


if __name__ == "__main__":
    main()