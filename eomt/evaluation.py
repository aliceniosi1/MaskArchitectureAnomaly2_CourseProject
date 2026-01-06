# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
import random
import sys

import cv2  # non usato direttamente, ma lo lasciamo per uniformità
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
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

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 19              # Cityscapes train IDs
IMG_SIZE = (1024, 1024)       # img_size usato nel data module CityscapesSemantic
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize(IMG_SIZE, Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform = Compose(
    [
        Resize(IMG_SIZE, Image.NEAREST),
    ]
)

# -----------------------------------------------------------------------------
# METRICA FPR@95TPR (al posto di ood_metrics.fpr_at_95_tpr)
# -----------------------------------------------------------------------------
def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcola FPR@95%TPR dato un vettore di punteggi 'scores'
    e un vettore di label binarie 'labels' (1 = OOD, 0 = in-distribution).
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)

    # indici dove TPR >= 0.95
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        # se non raggiungiamo mai 95% di TPR, restituiamo FPR massimo
        return 1.0

    # primo threshold che raggiunge almeno 95% TPR
    return float(fpr[idxs[0]])


# -----------------------------------------------------------------------------
# COSTRUZIONE E CARICAMENTO DEL MODELLO EoMT
# -----------------------------------------------------------------------------
def load_eomt_model(ckpt_path: str, device: torch.device) -> EoMT:
    """
    Costruisce il backbone ViT + EoMT e carica i pesi da un checkpoint
    tramite la LightningModule del progetto.
    """
    # Encoder DINOv2
    encoder = ViT(
        img_size=IMG_SIZE,
        backbone_name=BACKBONE_NAME,
    )

    # Rete EoMT
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=True,
    )

    # Usiamo LightningModule solo per caricare comodamente il checkpoint
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
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# FUNZIONE PER CALCOLARE LA MAPPA DI ANOMALIA
# -----------------------------------------------------------------------------
def compute_anomaly_map_from_logits(logits: torch.Tensor, method: str) -> torch.Tensor:
    """
    logits: Tensor [C, H, W]
    method: 'msp' | 'maxlogit' | 'maxentropy' | 'rba'
    ritorna una mappa [H, W] dove valori alti = più anomalo
    """
    probs = F.softmax(logits, dim=0)  # softmax sui canali C

    if method == "msp":
        # Maximum Softmax Probability → anomalia = 1 - MSP
        msp = probs.max(dim=0).values
        anomaly = 1.0 - msp
    elif method == "maxlogit":
        # anomalia alta quando il logit massimo è basso
        maxlogit = logits.max(dim=0).values
        anomaly = -maxlogit
    elif method == "maxentropy":
        eps = 1e-8
        entropy = -(probs * (probs + eps).log()).sum(dim=0)
        anomaly = entropy
    elif method == "rba":
        # Versione semplificata stile “reject-based”:
        # pixel con MSP sotto soglia → più anomali
        msp = probs.max(dim=0).values
        accept_threshold = 0.5
        anomaly = torch.clamp(accept_threshold - msp, min=0) / accept_threshold
    else:
        raise ValueError(f"Unknown method: {method}")

    return anomaly


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument("--loadDir", default="../trained_models/")
    parser.add_argument(
        "--loadWeights",
        default="eomt_cityscapes.bin",  # nome del file di pesi EoMT
        help="Checkpoint EoMT pre-addestrato su Cityscapes",
    )
    parser.add_argument(
        "--method",
        default="msp",
        choices=["msp", "maxlogit", "maxentropy", "rba"],
        help="Metodo per generare la mappa di anomalia",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    anomaly_score_list = []
    ood_gts_list = []

    # Device
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    file = open("results.txt", "a")

    ckpt_path = os.path.join(args.loadDir, args.loadWeights)

    print("Loading EoMT checkpoint:", ckpt_path)
    model = load_eomt_model(ckpt_path, device)
    print("Model LOADED successfully (EoMT + DINOv2)")
    model.eval()

    # -------------------------------------------------------------------------
    # LOOP sulle immagini
    # -------------------------------------------------------------------------
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)

        # Immagine → tensor [1, 3, H, W]
        img_pil = Image.open(path).convert("RGB")
        images = input_transform(img_pil).unsqueeze(0).float().to(device)

        with torch.no_grad():
            # EoMT ritorna liste per-layer
            mask_logits_per_layer, class_logits_per_layer = model(images)

            # Usiamo SOLO l'ultimo layer (come in training)
            mask_logits = mask_logits_per_layer[-1]      # [B, Q, h, w]
            class_logits = class_logits_per_layer[-1]    # [B, Q, C+1]

            # Ridimensioniamo le mask alla risoluzione target (IMG_SIZE)
            mask_logits = F.interpolate(
                mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False
            )

            # Logits per pixel [B, C, H, W]
            per_pixel_logits = LightningModule.to_per_pixel_logits_semantic(
                mask_logits, class_logits
            )
            logits = per_pixel_logits[0]  # [C, H, W]

            # Mappa di anomalia [H, W]
            anomaly_tensor = compute_anomaly_map_from_logits(logits, args.method)
            anomaly_result = anomaly_tensor.detach().cpu().numpy()

        # ---------------------------------------------------------------------
        # Ground truth OOD (stesso codice dell'eval ERFNet)
        # ---------------------------------------------------------------------
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # Skippa immagini senza pixel OOD
        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)

        del mask, ood_gts, anomaly_result, logits, per_pixel_logits
        torch.cuda.empty_cache()

    file.write("\n")

    # -------------------------------------------------------------------------
    # Calcolo metriche (identico allo script ERFNet)
    # -------------------------------------------------------------------------
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f"[EoMT-{args.method}] AUPRC score: {prc_auc * 100.0}")
    print(f"[EoMT-{args.method}] FPR@TPR95: {fpr * 100.0}")

    file.write(
        f"    [EoMT-{args.method}] AUPRC score:{prc_auc * 100.0}"
        f"   FPR@TPR95:{fpr * 100.0}"
    )
    file.close()


if __name__ == "__main__":
    main()