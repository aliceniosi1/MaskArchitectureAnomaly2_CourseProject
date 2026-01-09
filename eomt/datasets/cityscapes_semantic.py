# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms.v2.functional import InterpolationMode

from datasets.coco_instance import COCOInstance
from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

# --- Outlier Exposure (OE) wrapper for segmentation datasets ---
class CutPasteOEDataset(torch.utils.data.Dataset):
    """Wrap a base segmentation dataset and paste a random COCO instance into the image.

    Returns the same (img, target) format as the base dataset, but adds:
      target["oe_mask"]: tv_tensors.Mask[H,W] (bool) marking pasted (OOD) pixels.

    Additionally, base GT masks are cleaned in the pasted region so the ID loss does not
    supervise occluded pixels.
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        coco_dataset: torch.utils.data.Dataset,
        img_size: tuple[int, int],
        paste_prob: float = 0.5,
        min_rel_size: float = 0.05,
        max_rel_size: float = 0.35,
        max_tries: int = 20,
    ):
        self.base_dataset = base_dataset
        self.coco_dataset = coco_dataset
        self.img_size = img_size
        self.paste_prob = float(paste_prob)
        self.min_rel_size = float(min_rel_size)
        self.max_rel_size = float(max_rel_size)
        self.max_tries = int(max_tries)

        if not (0.0 <= self.paste_prob <= 1.0):
            raise ValueError("paste_prob must be in [0,1]")
        if self.min_rel_size <= 0 or self.max_rel_size <= 0:
            raise ValueError("min_rel_size and max_rel_size must be > 0")
        if self.min_rel_size > self.max_rel_size:
            raise ValueError("min_rel_size must be <= max_rel_size")

    def __len__(self):
        return len(self.base_dataset)

    @staticmethod
    def _bbox_from_mask(mask_hw: torch.Tensor):
        ys, xs = torch.where(mask_hw)
        if ys.numel() == 0:
            return None
        y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
        x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
        return y0, x0, y1, x1

    def __getitem__(self, idx: int):
        img, target = self.base_dataset[idx]

        # Only apply OE with probability paste_prob
        if torch.rand(()) >= self.paste_prob:
            return img, target

        # Expect image in [C,H,W]
        H, W = int(img.shape[-2]), int(img.shape[-1])
        oe_mask_full = torch.zeros((H, W), dtype=torch.bool)

        # Try multiple times to get a valid COCO instance
        for _ in range(self.max_tries):
            coco_idx = int(torch.randint(0, len(self.coco_dataset), (1,)).item())
            coco_img, coco_target = self.coco_dataset[coco_idx]

            coco_masks = coco_target.get("masks", None)
            if coco_masks is None or coco_masks.numel() == 0:
                continue

            n_inst = int(coco_masks.shape[0])
            inst_idx = int(torch.randint(0, n_inst, (1,)).item())
            inst_mask = coco_masks[inst_idx].bool()

            bbox = self._bbox_from_mask(inst_mask)
            if bbox is None:
                continue

            y0, x0, y1, x1 = bbox
            if (y1 - y0) < 8 or (x1 - x0) < 8:
                continue

            # Crop object patch
            patch_img = TF.crop(coco_img, y0, x0, y1 - y0, x1 - x0)
            patch_mask = TF.crop(inst_mask, y0, x0, y1 - y0, x1 - x0)

            # Random resize based on relative size of the Cityscapes crop
            rel = float(torch.empty(1).uniform_(self.min_rel_size, self.max_rel_size).item())
            target_max_side = max(8, int(rel * min(H, W)))
            ph, pw = int(patch_mask.shape[-2]), int(patch_mask.shape[-1])
            scale = target_max_side / float(max(ph, pw))
            new_h = max(8, int(round(ph * scale)))
            new_w = max(8, int(round(pw * scale)))

            if new_h >= H or new_w >= W:
                continue

            patch_img = TF.resize(patch_img, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
            patch_mask = TF.resize(patch_mask.float(), [new_h, new_w], interpolation=InterpolationMode.NEAREST) > 0.5

            if patch_mask.sum() == 0:
                continue

            # Random placement
            top = int(torch.randint(0, H - new_h + 1, (1,)).item())
            left = int(torch.randint(0, W - new_w + 1, (1,)).item())

            # Paste
            region = img[:, top : top + new_h, left : left + new_w]
            patch_img_t = patch_img
            if patch_img_t.dtype != region.dtype:
                patch_img_t = patch_img_t.to(region.dtype)

            m = patch_mask
            region[:, m] = patch_img_t[:, m]
            img[:, top : top + new_h, left : left + new_w] = region

            oe_mask_full[top : top + new_h, left : left + new_w] |= m

            # Clean GT masks where OE occludes pixels
            masks = target["masks"]
            labels = target["labels"]
            is_crowd = target["is_crowd"]

            masks_clean = masks.clone()
            masks_clean[:, oe_mask_full] = False
            keep = masks_clean.flatten(1).any(1)

            # If everything becomes empty, skip OE for this sample
            if not bool(keep.any()):
                return img, target

            target_new = dict(target)
            target_new["masks"] = tv_tensors.Mask(masks_clean[keep], dtype=torch.bool)
            target_new["labels"] = labels[keep]
            target_new["is_crowd"] = is_crowd[keep]
            target_new["oe_mask"] = tv_tensors.Mask(oe_mask_full, dtype=torch.bool)

            return img, target_new

        # Fallback: no successful paste
        return img, target


class CityscapesSemantic(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
        # --- Step 0 (OE) options ---
        oe_enabled: bool = False,
        oe_coco_path: Optional[str] = None,
        oe_paste_prob: float = 0.5,
        oe_min_rel_size: float = 0.05,
        oe_max_rel_size: float = 0.35,
        oe_max_tries: int = 20,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

        # --- Step 0 (OE) ---
        self.oe_enabled = bool(oe_enabled)
        self.oe_coco_path = oe_coco_path
        self.oe_paste_prob = float(oe_paste_prob)
        self.oe_min_rel_size = float(oe_min_rel_size)
        self.oe_max_rel_size = float(oe_max_rel_size)
        self.oe_max_tries = int(oe_max_tries)

    @staticmethod
    def target_parser(target, **kwargs):
        masks, labels = [], []

        for label_id in target[0].unique():
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            if cls is None or cls.ignore_in_eval:
                continue

            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.path, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.path, "gtFine_trainvaltest.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        self.cityscapes_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            **cityscapes_dataset_kwargs,
        )

        # Optional Outlier Exposure (OE): paste random COCO instances into Cityscapes training images
        if self.oe_enabled:
            if not self.oe_coco_path:
                raise ValueError("oe_enabled=True but oe_coco_path is None")

            coco_root = Path(self.oe_coco_path)
            coco_train_dataset = Dataset(
                transforms=None,
                img_folder_path_in_zip=Path("./train2017"),
                annotations_json_path_in_zip=Path("./annotations/instances_train2017.json"),
                target_zip_path=Path(coco_root, "annotations_trainval2017.zip"),
                zip_path=Path(coco_root, "train2017.zip"),
                img_suffix=".jpg",
                target_parser=COCOInstance.target_parser,
                only_annotations_json=True,
                check_empty_targets=True,
            )

            self.cityscapes_train_dataset = CutPasteOEDataset(
                base_dataset=self.cityscapes_train_dataset,
                coco_dataset=coco_train_dataset,
                img_size=self.img_size,
                paste_prob=self.oe_paste_prob,
                min_rel_size=self.oe_min_rel_size,
                max_rel_size=self.oe_max_rel_size,
                max_tries=self.oe_max_tries,
            )

        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            **cityscapes_dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
