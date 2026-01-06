# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import glob
import random
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize, ToTensor

# -----------------------------------------------------------------------------
# SETUP PATH & IMPORTS
# -----------------------------------------------------------------------------
# Aggiungiamo dinamicamente la root del progetto al path per importare i moduli custom
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")

if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

from models.vit import ViT
from models.eomt import EoMT
from training.lightning_module import LightningModule

# -----------------------------------------------------------------------------
# CONFIGURAZIONE & PARAMETRI
# -----------------------------------------------------------------------------
SEED = 42
NUM_CLASSES = 19              # Classi Cityscapes
IMG_SIZE = (1024, 1024)       # Risoluzione input/output coerente col training
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

# Setup riproducibilità
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Trasformazioni Input/Target
input_transform = Compose([
    Resize(IMG_SIZE, Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize(IMG_SIZE, Image.NEAREST),
])

# -----------------------------------------------------------------------------
# FUNZIONI DI UTILITÀ
# -----------------------------------------------------------------------------

def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcola il False Positive Rate (FPR) quando il True Positive Rate (TPR) è al 95%.
    
    Args:
        scores: array dei punteggi di anomalia.
        labels: array binario delle etichette (1 = anomalia, 0 = in-distribution).
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    
    # Trova gli indici dove il TPR supera il 95%
    idxs = np.where(tpr >= 0.95)[0]
    
    if len(idxs) == 0:
        return 1.0  # Fallback pessimistico se non si raggiunge il target TPR

    return float(fpr[idxs[0]])

def load_eomt_model(ckpt_path: str, device: torch.device) -> EoMT:
    """
    Inizializza l'architettura EoMT (Backbone ViT + Decoder) e carica i pesi
    usando la logica del LightningModule originale.
    """
    print(f"--> Inizializzazione Backbone: {BACKBONE_NAME}")
    encoder = ViT(img_size=IMG_SIZE, backbone_name=BACKBONE_NAME)

    print(f"--> Inizializzazione EoMT Network (Classes: {NUM_CLASSES}, Queries: {NUM_QUERIES})")
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=True,
    )

    # Utilizziamo il LightningModule come wrapper per il caricamento sicuro dei pesi
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

def compute_anomaly_map(logits: torch.Tensor, method: str) -> torch.Tensor:
    """
    Genera una mappa di anomalia pixel-wise a partire dai logits semantici.
    
    Args:
        logits: Tensore [C, H, W]
        method: Strategia di scoring ('msp', 'maxlogit', 'maxentropy', 'rba')
    """
    probs = F.softmax(logits, dim=0)

    if method == "msp":
        # Maximum Softmax Probability: Score = 1 - max(P(y|x))
        msp = probs.max(dim=0).values
        anomaly_map = 1.0 - msp

    elif method == "maxlogit":
        # Max Logit: Score = -max(Logits)
        maxlogit = logits.max(dim=0).values
        anomaly_map = -maxlogit

    elif method == "maxentropy":
        # Entropia: Score = H(P(y|x))
        eps = 1e-8
        entropy = -(probs * (probs + eps).log()).sum(dim=0)
        anomaly_map = entropy

    elif method == "rba":
        # Reject-Based Acceptance (semplificato)
        #msp = probs.max(dim=0).values
        #accept_threshold = 0.5
        #anomaly_map = torch.clamp(accept_threshold - msp, min=0) / accept_threshold

        anomaly_map = -logits.tanh().sum(dim=0)

    else:
        raise ValueError(f"Metodo anomalia non supportato: {method}")

    return anomaly_map

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp", help="Path o glob pattern immagini input")
    parser.add_argument("--loadDir", default="../trained_models/", help="Cartella dei modelli salvati")
    parser.add_argument("--loadWeights", default="eomt_cityscapes.bin", help="Nome file checkpoint")
    parser.add_argument("--method", default="msp", choices=["msp", "maxlogit", "maxentropy", "rba"], help="Metodo score anomalia")
    parser.add_argument("--cpu", action="store_true", help="Forza esecuzione su CPU")
    args = parser.parse_args()

    # Configurazione Device
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device in uso: {device}")

    # Preparazione file output
    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    
    ckpt_full_path = os.path.join(args.loadDir, args.loadWeights)
    print(f"Caricamento pesi da: {ckpt_full_path}")

    # Caricamento Modello
    try:
        model = load_eomt_model(ckpt_full_path, device)
        print("Modello caricato con successo.")
    except Exception as e:
        print(f"Errore critico nel caricamento del modello: {e}")
        return

    # Liste per accumulo risultati
    anomaly_score_list = []
    ood_gts_list = []
    
    file_list = glob.glob(os.path.expanduser(str(args.input[0])))
    print(f"Trovate {len(file_list)} immagini da elaborare.")

    # -------------------------------------------------------------------------
    # INIZIO ELABORAZIONE BATCH
    # -------------------------------------------------------------------------
    for path in file_list:
        print(f"Processing: {path}")

        # 1. Inferenza
        img_pil = Image.open(path).convert("RGB")
        img_tensor = input_transform(img_pil).unsqueeze(0).float().to(device)

        with torch.no_grad():
            # EoMT forward pass
            mask_logits_layers, class_logits_layers = model(img_tensor)

            # Estrazione output dall'ultimo layer
            final_mask_logits = mask_logits_layers[-1]    # [B, Q, h, w]
            final_class_logits = class_logits_layers[-1]  # [B, Q, C+1]

            # Upsample alla risoluzione originale
            final_mask_logits = F.interpolate(
                final_mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False
            )

            # Conversione in Semantic Logits standard [B, C, H, W]
            per_pixel_logits = LightningModule.to_per_pixel_logits_semantic(
                final_mask_logits, final_class_logits
            )
            pixel_logits = per_pixel_logits[0] # Rimuovi batch dimension -> [C, H, W]

            # Calcolo Mappa Anomalia
            anomaly_map = compute_anomaly_map(pixel_logits, args.method)
            anomaly_np = anomaly_map.detach().cpu().numpy()

        # 2. Caricamento e Adattamento Ground Truth (Labels)
        pathGT = path.replace("images", "labels_masks")
        
        # Gestione estensioni file specifiche per dataset
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")

        try:
            gt_img = Image.open(pathGT)
        except FileNotFoundError:
            print(f"Warning: GT non trovata per {path}, skip.")
            continue

        gt_img = target_transform(gt_img)
        ood_gts = np.array(gt_img)

        # Mappatura Labels Dataset -> Formato Binario (0=In-Dist, 1=OOD, 255=Ignore)
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        elif "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
        elif "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # Se l'immagine non contiene pixel OOD validi, la saltiamo
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_np)

        # Pulizia memoria GPU
        del img_tensor, anomaly_map, pixel_logits
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # CALCOLO METRICHE FINALI
    # -------------------------------------------------------------------------
    if not ood_gts_list:
        print("Nessun dato valido raccolto per la valutazione.")
        return

    print("Calcolo metriche in corso...")
    
    # Flattening array per calcolo globale
    ood_gts_flat = np.array(ood_gts_list)
    anomaly_scores_flat = np.array(anomaly_score_list)

    # Maschere booleane
    ood_mask = (ood_gts_flat == 1)
    ind_mask = (ood_gts_flat == 0)

    # Estrazione punteggi
    ood_scores = anomaly_scores_flat[ood_mask]
    ind_scores = anomaly_scores_flat[ind_mask]

    # Creazione etichette per Sklearn
    all_scores = np.concatenate((ind_scores, ood_scores))
    all_labels = np.concatenate((np.zeros(len(ind_scores)), np.ones(len(ood_scores))))

    # Metriche
    auprc = average_precision_score(all_labels, all_scores)
    fpr95 = fpr_at_95_tpr(all_scores, all_labels)

    result_str = f"[EoMT-{args.method}] AUPRC: {auprc * 100.0:.2f}% | FPR@95TPR: {fpr95 * 100.0:.2f}%"
    print("\n" + "="*50)
    print(result_str)
    print("="*50 + "\n")

    # Scrittura su file
    with open("results.txt", "a") as f:
        f.write("\n" + result_str)

if __name__ == "__main__":
    main()