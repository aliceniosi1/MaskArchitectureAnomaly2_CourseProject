# msp_histograms.py
# Produce MSP score histograms (IND vs OOD) for RA-21 and FS L&F,
# comparing baseline vs extension checkpoints.

import os
import glob
import random
import sys
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# IMPORT EoMT (aggiungo la cartella eomt al PYTHONPATH) - stesso schema del tuo evaluation.py
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
# CONFIG BASE (allineati al tuo file)
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_CLASSES = 19
IMG_SIZE = (1024, 1024)  # (H,W) -> come evaluation.py
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"
IGNORE_LABEL = 255

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize(IMG_SIZE, Image.BILINEAR),
        ToTensor(),
    ]
)

# -----------------------------------------------------------------------------
# MODEL LOADER (uguale concetto del tuo)
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
# GT loader: {0,1,255} invariato, RoadAnomaly {0,2}->{0,1} (come evaluation.py)
# -----------------------------------------------------------------------------
def load_ood_gt_from_img_path(img_path: str, out_size=None) -> np.ndarray:
    pathGT = img_path.replace("images", "labels_masks")
    pathGT = os.path.splitext(pathGT)[0] + ".png"

    gt_pil = Image.open(pathGT).convert("L")
    if out_size is not None:
        gt_pil = gt_pil.resize((out_size[1], out_size[0]), resample=Image.NEAREST)

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
# MSP map (stesso spirito del tuo: pixel_scores -> softmax -> 1-max)
# -----------------------------------------------------------------------------
@torch.no_grad()
def pixel_scores_with_temperature(
    mask_logits: torch.Tensor,   # [B,Q,H,W]
    class_logits: torch.Tensor,  # [B,Q,C+1]
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    mask_probs = mask_logits.sigmoid()
    class_probs = torch.softmax(class_logits / temperature, dim=-1)[..., :-1]  # drop void/no-object

    # [B,Q,H,W] x [B,Q,C] -> [B,C,H,W]
    pixel_scores = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)
    return pixel_scores

@torch.no_grad()
def msp_anomaly_map(pixel_scores_chw: torch.Tensor) -> torch.Tensor:
    # Nota: replica la tua scelta (softmax lungo C) anche se pixel_scores sono già “pseudo-prob”
    probs = torch.softmax(pixel_scores_chw, dim=0)  # [C,H,W]
    return 1.0 - probs.max(dim=0).values            # [H,W]

# -----------------------------------------------------------------------------
# Raccolta campioni IND/OOD
# -----------------------------------------------------------------------------
def list_images(patterns: List[str]) -> List[str]:
    img_paths: List[str] = []
    for pat in patterns:
        img_paths.extend(glob.glob(os.path.expanduser(str(pat))))
    img_paths = sorted(list(set(img_paths)))
    if not img_paths:
        raise FileNotFoundError(f"No images matched input patterns: {patterns}")
    return img_paths

@torch.no_grad()
def collect_ind_ood_scores(
    model: EoMT,
    img_paths: List[str],
    device: torch.device,
    temperature: float,
    sample_ind_per_image: int,
    sample_ood_per_image: int,
    only_images_with_ood: bool = True,
    max_images: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)

    if max_images > 0:
        img_paths = img_paths[:max_images]

    ind_scores: List[np.ndarray] = []
    ood_scores: List[np.ndarray] = []

    processed = 0
    used = 0

    for path in img_paths:
        processed += 1

        img_pil = Image.open(path).convert("RGB")
        images = input_transform(img_pil).unsqueeze(0).float().to(device)  # [1,3,H,W]

        gt = load_ood_gt_from_img_path(path, out_size=IMG_SIZE)            # [H,W]
        valid = (gt <= 1)

        has_ood = bool((gt == 1).any())
        if only_images_with_ood and not has_ood:
            continue

        mask_logits_per_layer, class_logits_per_layer = model(images)
        mask_logits = mask_logits_per_layer[-1]
        class_logits = class_logits_per_layer[-1]

        mask_logits = F.interpolate(mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False)

        pix_scores = pixel_scores_with_temperature(mask_logits, class_logits, temperature=temperature)[0]  # [C,H,W]
        amap = msp_anomaly_map(pix_scores).detach().cpu().numpy().astype(np.float32, copy=False)          # [H,W]

        ind_idx = np.flatnonzero(valid & (gt == 0))
        ood_idx = np.flatnonzero(valid & (gt == 1))

        if ind_idx.size > 0:
            k = min(sample_ind_per_image, ind_idx.size)
            pick = rng.choice(ind_idx, size=k, replace=False)
            ind_scores.append(amap.reshape(-1)[pick])

        if ood_idx.size > 0:
            k = min(sample_ood_per_image, ood_idx.size)
            # per OOD spesso sono pochi: prendo tutti se < k
            if k == ood_idx.size:
                pick = ood_idx
            else:
                pick = rng.choice(ood_idx, size=k, replace=False)
            ood_scores.append(amap.reshape(-1)[pick])

        used += 1

        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if used % 25 == 0:
            print(f"[collect] used={used} processed={processed}/{len(img_paths)}")

    if len(ind_scores) == 0 or len(ood_scores) == 0:
        raise RuntimeError(
            f"Not enough samples collected. ind_chunks={len(ind_scores)} ood_chunks={len(ood_scores)} "
            f"(maybe patterns/GT path mismatch?)"
        )

    ind = np.concatenate(ind_scores, axis=0)
    ood = np.concatenate(ood_scores, axis=0)
    return ind, ood

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_compare_hist(
    dataset_name: str,
    baseline_ind: np.ndarray,
    baseline_ood: np.ndarray,
    ext_ind: np.ndarray,
    ext_ood: np.ndarray,
    out_path: str,
    bins: int = 60,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    # Baseline
    axes[0].hist(baseline_ind, bins=bins, density=True, alpha=0.6, label="IND")
    axes[0].hist(baseline_ood, bins=bins, density=True, alpha=0.6, label="OOD")
    axes[0].set_title("Baseline")
    axes[0].set_xlabel("MSP anomaly score (1 - max prob)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Extension
    axes[1].hist(ext_ind, bins=bins, density=True, alpha=0.6, label="IND")
    axes[1].hist(ext_ood, bins=bins, density=True, alpha=0.6, label="OOD")
    axes[1].set_title("Extension (LogitNorm)")
    axes[1].set_xlabel("MSP anomaly score (1 - max prob)")
    axes[1].legend()

    fig.suptitle(f"{dataset_name}: MSP score distributions (IND vs OOD)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    p = ArgumentParser()
    p.add_argument("--ra21", nargs="+", required=True, help='Glob RA-21 images, es ".../RA-21/images/*.jpg"')
    p.add_argument("--fslf", nargs="+", required=True, help='Glob FS L&F images, es ".../fs_lost_found/images/*.jpg"')

    p.add_argument("--baseline_dir", required=True)
    p.add_argument("--baseline_weights", required=True)

    p.add_argument("--ext_dir", required=True)
    p.add_argument("--ext_weights", required=True)

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample_ind_per_image", type=int, default=8000)
    p.add_argument("--sample_ood_per_image", type=int, default=8000)
    p.add_argument("--only_images_with_ood", action="store_true", help="Usa solo immagini con almeno 1 pixel OOD")
    p.add_argument("--max_images", type=int, default=-1, help="Debug: limita numero immagini")
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--out_dir", type=str, default="./msp_hists", help="Cartella output per PNG")
    p.add_argument("--bins", type=int, default=60)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    baseline_ckpt = os.path.join(args.baseline_dir, args.baseline_weights)
    ext_ckpt = os.path.join(args.ext_dir, args.ext_weights)
    print("Baseline ckpt:", baseline_ckpt)
    print("Extension ckpt:", ext_ckpt)

    baseline_model = load_eomt_model(baseline_ckpt, device)
    ext_model = load_eomt_model(ext_ckpt, device)

    ra21_paths = list_images(args.ra21)
    fslf_paths = list_images(args.fslf)

    # --- RA-21 ---
    print("\n[RA-21] Collecting baseline...")
    b_ind, b_ood = collect_ind_ood_scores(
        baseline_model, ra21_paths, device,
        temperature=args.temperature,
        sample_ind_per_image=args.sample_ind_per_image,
        sample_ood_per_image=args.sample_ood_per_image,
        only_images_with_ood=args.only_images_with_ood,
        max_images=args.max_images,
    )
    print("[RA-21] Collecting extension...")
    e_ind, e_ood = collect_ind_ood_scores(
        ext_model, ra21_paths, device,
        temperature=args.temperature,
        sample_ind_per_image=args.sample_ind_per_image,
        sample_ood_per_image=args.sample_ood_per_image,
        only_images_with_ood=args.only_images_with_ood,
        max_images=args.max_images,
    )

    out_ra21 = os.path.join(args.out_dir, "RA21_msp_hist_compare.png")
    plot_compare_hist("SMIYC RA-21", b_ind, b_ood, e_ind, e_ood, out_ra21, bins=args.bins)
    print("Saved:", out_ra21)

    # --- FS L&F ---
    print("\n[FS L&F] Collecting baseline...")
    b_ind, b_ood = collect_ind_ood_scores(
        baseline_model, fslf_paths, device,
        temperature=args.temperature,
        sample_ind_per_image=args.sample_ind_per_image,
        sample_ood_per_image=args.sample_ood_per_image,
        only_images_with_ood=args.only_images_with_ood,
        max_images=args.max_images,
    )
    print("[FS L&F] Collecting extension...")
    e_ind, e_ood = collect_ind_ood_scores(
        ext_model, fslf_paths, device,
        temperature=args.temperature,
        sample_ind_per_image=args.sample_ind_per_image,
        sample_ood_per_image=args.sample_ood_per_image,
        only_images_with_ood=args.only_images_with_ood,
        max_images=args.max_images,
    )

    out_fslf = os.path.join(args.out_dir, "FSLF_msp_hist_compare.png")
    plot_compare_hist("Fishyscapes L&F", b_ind, b_ood, e_ind, e_ood, out_fslf, bins=args.bins)
    print("Saved:", out_fslf)

if __name__ == "__main__":
    main()