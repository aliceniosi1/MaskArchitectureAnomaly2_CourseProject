# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        # --- Step 0 (OE) options ---
        oe_enabled: bool = False,
        oe_lambda: float = 0.1,
        oe_temperature: float = 1.0,
        train_class_head_only: bool = False,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            train_class_head_only=train_class_head_only,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        # --- Step 0 (OE) ---
        self.oe_enabled = oe_enabled
        self.oe_lambda = float(oe_lambda)
        self.oe_temperature = float(oe_temperature)
        if self.oe_temperature <= 0:
            raise ValueError("oe_temperature must be > 0")

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_semantic(ignore_idx, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

    def training_step(self, batch, batch_idx):
        """Training step with optional Outlier Exposure (OE).

        Expected batch format (from datamodule):
          imgs: Tensor [B,3,H,W]
          targets: list[dict] with keys {"masks", "labels", ...}
        If OE is enabled, each target may additionally contain:
          target["oe_mask"]: Bool Tensor [H,W] marking pasted (OOD) pixels.
        """
        imgs, targets = batch

        # Forward once
        mask_logits_per_block, class_logits_per_block = self(imgs)

        # Standard Mask2Former-style losses (all blocks)
        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        loss_total = self.criterion.loss_total(losses_all_blocks, self.log)

        # Optional OE loss: encourage high-entropy / near-uniform predictions on pasted pixels
        if self.oe_enabled:
            # Collect oe masks if present
            oe_masks = [t.get("oe_mask", None) for t in targets]
            has_any_oe = any(m is not None for m in oe_masks)

            if has_any_oe:
                # Use the last block for OE regularization
                mask_logits = mask_logits_per_block[-1]
                class_logits = class_logits_per_block[-1]

                # Upscale masks to image resolution
                mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")

                # Build per-pixel class scores then normalize across classes
                pixel_scores = self.to_per_pixel_logits_semantic(mask_logits, class_logits)  # [B,C,H,W]
                probs = torch.softmax(pixel_scores / self.oe_temperature, dim=1)
                log_probs = torch.log(probs.clamp_min(1e-8))

                oe_loss_sum = None
                oe_count = 0

                for b, m in enumerate(oe_masks):
                    if m is None:
                        continue
                    # Ensure bool [H,W]
                    if m.ndim == 3 and m.shape[0] == 1:
                        m = m[0]
                    m = m.to(device=log_probs.device).bool()

                    # If mask resolution mismatches, resize with nearest
                    if tuple(m.shape[-2:]) != tuple(self.img_size):
                        m = (
                            F.interpolate(
                                m[None, None].float(),
                                size=self.img_size,
                                mode="nearest",
                            )[0, 0]
                            > 0.5
                        )

                    if m.sum() == 0:
                        continue

                    # Cross-entropy to uniform distribution over known classes:
                    # L = -mean_c log p_c  (averaged over classes and selected pixels)
                    oe_loss_b = -log_probs[b][:, m].mean()

                    oe_loss_sum = oe_loss_b if oe_loss_sum is None else (oe_loss_sum + oe_loss_b)
                    oe_count += 1

                if oe_count > 0:
                    oe_loss = oe_loss_sum / oe_count  # type: ignore
                    self.log("losses/train_oe_uniform", oe_loss, sync_dist=True)
                    loss_total = loss_total + (self.oe_lambda * oe_loss)

        return loss_total

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        self._on_eval_end_semantic("val")
