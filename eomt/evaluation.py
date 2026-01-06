import os, glob, random, sys, hashlib
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score, roc_curve

# -----------------------------------------------------------------------------
# IMPORT EoMT
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")
if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

from models.vit import ViT
from models.eomt import EoMT
from training.lightning_module import LightningModule

# -----------------------------------------------------------------------------
# Utils (come i tuoi)
# -----------------------------------------------------------------------------
def try_paths(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def infer_gt_path(img_path: str) -> str:
    base = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels_masks" + os.sep)
    candidates = [
        base,
        os.path.splitext(base)[0] + ".png",
        os.path.splitext(base)[0] + ".jpg",
        os.path.splitext(base)[0] + ".jpeg",
        os.path.splitext(base)[0] + ".webp",
    ]
    gt = try_paths(candidates)
    if gt is None:
        raise FileNotFoundError(f"GT not found for image: {img_path}\nTried:\n" + "\n".join(candidates))
    return gt

def build_ood_gt(gt_path: str, img_size, target_transform) -> np.ndarray:
    mask = Image.open(gt_path)
    mask = target_transform(mask)
    ood_gts = np.array(mask)
    p = gt_path.lower()

    if ("roadanomaly" in p) and ("roadanomaly21" not in p):
        ood_gts = np.where(ood_gts == 2, 1, 0).astype(np.uint8)

    elif ("roadanomaly21" in p) or ("roadobsticle21" in p) or ("roadobstacle21" in p) or ("fs_static" in p) or ("fs_lostfound_full" in p):
        ignore = (ood_gts == 255)
        ood_gts = np.where(ignore, 255, np.where(ood_gts == 1, 1, 0)).astype(np.uint8)

    else:
        ignore = (ood_gts == 255)
        ood_gts = np.where(ignore, 255, np.where(ood_gts == 1, 1, 0)).astype(np.uint8)

    # img_size = (H,W); numpy is (H,W); PIL resize uses (W,H) ma qui hai già fatto Resize giusto
    if ood_gts.shape[0] != img_size[0] or ood_gts.shape[1] != img_size[1]:
        raise ValueError(f"GT size mismatch. Got {ood_gts.shape}, expected (H,W)=({img_size[0]},{img_size[1]}) for {gt_path}")

    return ood_gts

# -----------------------------------------------------------------------------
# Model loader (come il tuo)
# -----------------------------------------------------------------------------
def load_eomt_model(
    ckpt_path: str,
    device: torch.device,
    img_size=(1024, 1024),
    num_classes=19,
    num_queries=100,
    num_blocks=3,
    backbone_name="vit_base_patch14_reg4_dinov2",
) -> EoMT:
    encoder = ViT(img_size=img_size, backbone_name=backbone_name)
    network = EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=num_queries,
        num_blocks=num_blocks,
        masked_attn_enabled=True,
    )

    lm = LightningModule(
        network=network,
        img_size=img_size,
        num_classes=num_classes,
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
# Combine + anomaly scores
# -----------------------------------------------------------------------------
def combine_to_per_pixel_scores(mask_logits_raw, class_logits, out_hw):
    """
    mask_logits_raw: (Q, h, w)   logits
    class_logits:    (Q, C+1)    logits
    out_hw: (H,W)

    returns scores: (C, H, W) non-negative "class score maps"
    """
    # upsample mask logits to (H,W)
    mask = F.interpolate(mask_logits_raw[None, ...], size=out_hw, mode="bilinear", align_corners=False)[0]  # (Q,H,W)

    # mask probs and class probs
    mask_p = mask.sigmoid()  # (Q,H,W)
    cls_p = class_logits.softmax(dim=-1)[..., :-1]  # (Q,C) (drop "no object")

    # scores_c(h,w) = sum_q mask_p(q,h,w) * cls_p(q,c)
    scores = torch.einsum("qhw,qc->chw", mask_p, cls_p)  # (C,H,W)
    return scores

def anomaly_msp(scores_chw, eps=1e-12):
    """MSP anomaly score computed from per-pixel class *logits*.

    In the official notebook flow, MSP is defined as:
        score = 1 - max_c softmax(logits)_c

    Here `scores_chw` is the per-pixel class score map produced by the
    MaskFormer-style combination of mask and class predictions.
    We treat it as logits and apply a Softmax across classes.
    """
    p = torch.softmax(scores_chw, dim=0)
    return 1.0 - p.max(dim=0).values  # (H,W)

def anomaly_maxentropy(scores_chw, eps=1e-12):
    """MaxEntropy anomaly score.

    Defined as the entropy of the Softmax probabilities:
        score = - sum_c p_c log(p_c),   p = softmax(logits)

    Higher entropy => more uncertainty => more likely OOD.
    """
    p = torch.softmax(scores_chw, dim=0)
    return -(p * (p + eps).log()).sum(dim=0)

def anomaly_maxlogit(scores_chw):
    # "logit" qui non è un logit puro: è un class-score map; lo uso come proxy
    return -scores_chw.max(dim=0).values

# -----------------------------------------------------------------------------
# Metrics (AuPRC e FPR@95)
# -----------------------------------------------------------------------------
def auprc_sklearn(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUPRC using sklearn (matches many reference eval scripts)."""
    y_true = (y_true > 0).astype(np.uint8)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def fpr_at_95_tpr_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """FPR@95%TPR computed from the ROC curve (matches reference script).

    `labels`: 1 = OOD, 0 = IND
    `scores`: higher => more OOD
    """
    labels = (labels > 0).astype(np.uint8)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0
    return float(fpr[idxs[0]])

def flatten_valid(score_hw: torch.Tensor, gt_hw: np.ndarray):
    """
    gt_hw: np array (H,W) in {0,1,255}
    """
    gt = gt_hw.reshape(-1)
    score = score_hw.detach().cpu().reshape(-1).numpy().astype(np.float64)
    valid = gt != 255
    return score[valid], gt[valid].astype(np.uint8)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True,
                        help="Glob pattern for images, e.g. '/path/RoadAnomaly/images/*.jpg'")
    parser.add_argument("--loadDir", default="../trained_models/")
    parser.add_argument("--loadWeights", default="eomt_cityscapes.bin")
    parser.add_argument("--img_size", default="1024,1024",
                        help="Model/GT size as 'H,W' (default 1024,1024).")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument(
        "--skip_no_ood",
        action="store_true",
        help="Skip images that contain no OOD pixels (matches some reference eval scripts).",
    )
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    H, W = [int(x.strip()) for x in args.img_size.split(",")]
    IMG_SIZE = (H, W)
    PIL_SIZE = (W, H)

    input_transform = Compose([Resize(PIL_SIZE, Image.BILINEAR), ToTensor()])
    target_transform = Compose([Resize(PIL_SIZE, Image.NEAREST)])

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt_path = os.path.join(args.loadDir, args.loadWeights)
    print("Loading EoMT checkpoint:", ckpt_path)
    model = load_eomt_model(ckpt_path, device, img_size=IMG_SIZE)

    # Build list of image paths
    img_paths = []
    for pat in args.input:
        img_paths.extend(glob.glob(os.path.expanduser(str(pat))))
    img_paths = sorted(list(set(img_paths)))
    if not img_paths:
        raise FileNotFoundError(f"No images matched input: {args.input}")
    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    # Accumulators
    all_gt = []
    all_msp = []
    all_ent = []
    all_mlog = []

    errors = 0

    for idx, path in enumerate(img_paths, 1):
        try:
            gt_path = infer_gt_path(path)
            ood_gts = build_ood_gt(gt_path, IMG_SIZE, target_transform)  # (H,W) in {0,1,255}

            img_pil = Image.open(path).convert("RGB")
            images = input_transform(img_pil).unsqueeze(0).float().to(device)

            with torch.no_grad():
                mask_logits_per_layer, class_logits_per_layer = model(images)
                mask_logits_raw = mask_logits_per_layer[-1][0]     # (Q,h,w)
                class_logits     = class_logits_per_layer[-1][0]   # (Q,C+1)

                scores_chw = combine_to_per_pixel_scores(mask_logits_raw, class_logits, out_hw=IMG_SIZE)  # (C,H,W)

                msp  = anomaly_msp(scores_chw)
                ment = anomaly_maxentropy(scores_chw)
                mlog = anomaly_maxlogit(scores_chw)

            msp_s,  gt_v = flatten_valid(msp,  ood_gts)

            # Optional: mimic reference scripts that skip images with no OOD pixels
            if args.skip_no_ood and (gt_v == 1).sum() == 0:
                continue

            ent_s,  _    = flatten_valid(ment, ood_gts)
            mlog_s, _    = flatten_valid(mlog, ood_gts)

            all_gt.append(gt_v)
            all_msp.append(msp_s)
            all_ent.append(ent_s)
            all_mlog.append(mlog_s)

            if idx % 10 == 0:
                print(f"[{idx}/{len(img_paths)}] ok (errors={errors})")

            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            print(f"ERROR on {path}: {e}")

    # Dataset-level metrics
    y = np.concatenate(all_gt, axis=0)
    msp_scores  = np.concatenate(all_msp, axis=0)
    ent_scores  = np.concatenate(all_ent, axis=0)
    mlog_scores = np.concatenate(all_mlog, axis=0)

    msp_auprc  = auprc_sklearn(y, msp_scores) * 100.0
    msp_fpr95  = fpr_at_95_tpr_roc(msp_scores, y) * 100.0
    ent_auprc  = auprc_sklearn(y, ent_scores) * 100.0
    ent_fpr95  = fpr_at_95_tpr_roc(ent_scores, y) * 100.0
    mlog_auprc = auprc_sklearn(y, mlog_scores) * 100.0
    mlog_fpr95 = fpr_at_95_tpr_roc(mlog_scores, y) * 100.0

    print("\n=== RESULTS ===")
    print(f"{'Method':<12} | {'AuPRC (%)':>10} | {'FPR@95 (%)':>10}")
    print("-" * 40)
    print(f"{'MSP':<12} | {msp_auprc:10.2f} | {msp_fpr95:10.2f}")
    print(f"{'MaxLogit':<12} | {mlog_auprc:10.2f} | {mlog_fpr95:10.2f}")
    print(f"{'MaxEntropy':<12} | {ent_auprc:10.2f} | {ent_fpr95:10.2f}")
    print(f"\nErrors: {errors} / {len(img_paths)}")

if __name__ == "__main__":
    main()