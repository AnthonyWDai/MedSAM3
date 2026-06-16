"""
SAM3 LoRA Training Script (CLI-only config)

Validation Strategy (Following SAM3):
  - During training: Only compute validation LOSS (fast, no metrics)
  - After training: Run validate_sam3_lora.py for full metrics (mAP, cgF1) with NMS

Examples:
  Single GPU:
    python train_sam3_petct.py \
      --data_dir /workspace/data \
      --output_dir outputs/sam3_lora_full \
      --device 0

  Multi-GPU:
    python train_sam3_petct.py \
      --data_dir /workspace/data \
      --output_dir outputs/sam3_lora_full \
      --device 0 1
"""

import os
import csv
import sys
import json
import argparse
import random
import shutil
import contextlib
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.transforms import v2
import pycocotools.mask as mask_utils

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from sam3.train.masks_ops import rle_encode

from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get the number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recommended_num_workers(reserve=1, train=True):
    if hasattr(os, "sched_getaffinity"):
        n_cpu = len(os.sched_getaffinity(0))
    else:
        n_cpu = os.cpu_count() or 1

    usable = max(1, n_cpu - reserve)

    if train:
        return max(1, min(usable, 4))
    else:
        return max(1, min(usable // 2, 2))


# ============================================================================
# Dataset
# ============================================================================

class FolderSegmentDataset(Dataset):
    """
    Dataset for segmentation data with text prompts read from CSV.

    Expected CSV columns:
        split,label,case_id,channel_0000,channel_0001,response

    Notes:
    - response is used as query_text
    - channel_0000 and channel_0001 are ignored for text prompt construction
    """

    def __init__(self, data_dir, csv_path, split="train"):
        self.data_dir = Path(data_dir)
        self.csv_path = Path(csv_path)
        self.split = split

        self.split_dir = self.data_dir / split
        self.images_dir = self.split_dir / "images"
        self.masks_dir = self.split_dir / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.samples = []

        valid_img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            sample_id = 0

            for row in reader:
                if row["split"] != split:
                    continue

                label = row["label"]
                case_id = row["case_id"]
                response = row["response"].strip() if row["response"] is not None else ""

                class_img_dir = self.images_dir / label
                class_mask_dir = self.masks_dir / label

                if not class_img_dir.exists():
                    print(f"Warning: image directory missing for label '{label}': {class_img_dir}")
                    continue
                if not class_mask_dir.exists():
                    print(f"Warning: mask directory missing for label '{label}': {class_mask_dir}")
                    continue

                # Find image by case_id stem, ignoring channel columns
                candidate_images = [
                    class_img_dir / f"{case_id}.png",
                    class_img_dir / f"{case_id}.jpg",
                    class_img_dir / f"{case_id}.jpeg",
                    class_img_dir / f"{case_id}.bmp",
                    class_img_dir / f"{case_id}.tif",
                    class_img_dir / f"{case_id}.tiff",
                ]
                img_path = next((p for p in candidate_images if p.exists()), None)

                candidate_masks = [
                    class_mask_dir / f"{case_id}.png",
                    class_mask_dir / f"{case_id}.jpg",
                    class_mask_dir / f"{case_id}.jpeg",
                    class_mask_dir / f"{case_id}.bmp",
                    class_mask_dir / f"{case_id}.tif",
                    class_mask_dir / f"{case_id}.tiff",
                ]
                mask_path = next((p for p in candidate_masks if p.exists()), None)

                if img_path is None:
                    print(f"Warning: no image found for case_id={case_id}, label={label}")
                    continue

                if mask_path is None:
                    print(f"Warning: no mask found for case_id={case_id}, label={label}")
                    continue

                self.samples.append({
                    "id": sample_id,
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "label": label,
                    "case_id": case_id,
                    "response": response,
                })
                sample_id += 1

        print(f"Loaded CSV-driven dataset: {split} split")
        print(f"  Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample["id"]
        img_path = sample["image_path"]
        mask_path = sample["mask_path"]
        response = sample["response"]

        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        resized_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)
        image_tensor = self.transform(resized_image)

        try:
            mask_image = PILImage.open(mask_path).convert("L")
            mask_np = np.array(mask_image)
            mask_bin = (mask_np > 0).astype(np.uint8)

            if mask_bin.sum() == 0:
                objects = []
                queries = [
                    FindQueryLoaded(
                        query_text=response,
                        image_id=0,
                        object_ids_output=[],
                        is_exhaustive=True,
                        query_processing_order=0,
                        inference_metadata=InferenceMetadata(
                            coco_image_id=img_id,
                            original_image_id=img_id,
                            original_category_id=0,
                            original_size=(orig_h, orig_w),
                            object_id=-1,
                            frame_index=-1
                        )
                    )
                ]
            else:
                ys, xs = np.where(mask_bin > 0)
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                x = float(x_min)
                y = float(y_min)
                w = float(x_max - x_min + 1)
                h = float(y_max - y_min + 1)

                cx = x + w / 2.0
                cy = y + h / 2.0

                scale_w = self.resolution / orig_w
                scale_h = self.resolution / orig_h

                box_tensor = torch.tensor([
                    cx * scale_w / self.resolution,
                    cy * scale_h / self.resolution,
                    w * scale_w / self.resolution,
                    h * scale_h / self.resolution,
                ], dtype=torch.float32)

                mask_t = torch.from_numpy(mask_bin).float().unsqueeze(0).unsqueeze(0)
                mask_t = torch.nn.functional.interpolate(
                    mask_t,
                    size=(self.resolution, self.resolution),
                    mode="nearest"
                )
                segment = mask_t.squeeze() > 0.5

                obj = Object(
                    bbox=box_tensor,
                    area=(box_tensor[2] * box_tensor[3]).item(),
                    object_id=0,
                    segment=segment
                )
                objects = [obj]

                queries = [
                    FindQueryLoaded(
                        query_text=response,
                        image_id=0,
                        object_ids_output=[0],
                        is_exhaustive=True,
                        query_processing_order=0,
                        inference_metadata=InferenceMetadata(
                            coco_image_id=img_id,
                            original_image_id=img_id,
                            original_category_id=0,
                            original_size=(orig_h, orig_w),
                            object_id=-1,
                            frame_index=-1
                        )
                    )
                ]

        except Exception as e:
            print(f"Warning: Error processing sample {img_path}: {e}")
            objects = []
            queries = [
                FindQueryLoaded(
                    query_text=response,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=0,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1
                    )
                )
            ]

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )
# ============================================================================
# Optional Eval Helpers (kept from original file)
# ============================================================================

def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.3):
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    used = torch.zeros(len(binary_masks), dtype=torch.bool)
    merged_masks, merged_scores, merged_boxes = [], [], []

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i].clone()
        used[i] = True

        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            intersection = (current_mask & binary_masks[j]).sum().float()
            union = (current_mask | binary_masks[j]).sum().float()
            iou = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def convert_predictions_to_coco_format(
    predictions_list,
    image_ids,
    resolution=288,
    score_threshold=0.0,
    merge_overlaps=True,
    iou_threshold=0.3,
    debug=False,
):
    coco_predictions = []
    pred_id = 0

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get("pred_logits", [])) == 0:
            continue

        logits = preds["pred_logits"]
        boxes = preds["pred_boxes"]
        masks = preds["pred_masks"]

        scores = torch.sigmoid(logits).squeeze(-1)

        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"  Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")

        binary_masks = (torch.sigmoid(masks) > 0.5).cpu()

        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"  Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"  Mask shape: {binary_masks.shape}")
                print(f"  Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for rle, score, box in zip(rles, scores.cpu().tolist(), boxes.cpu().tolist()):
                cx, cy, w, h = box
                x = (cx - w / 2) * resolution
                y = (cy - h / 2) * resolution
                w = w * resolution
                h = h * resolution

                coco_predictions.append({
                    "image_id": int(img_id),
                    "category_id": 1,
                    "segmentation": rle,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(score),
                    "id": pred_id,
                })
                pred_id += 1

    return coco_predictions


# ============================================================================
# Trainer
# ============================================================================

class SAM3TrainerNative:
    def __init__(self, args, multi_gpu=False):
        self.args = args

        set_seed(args.seed)

        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            device_name = args.hardware_device
            if device_name == "cuda" and not torch.cuda.is_available():
                device_name = "cpu"
            self.device = torch.device(device_name)

        print_rank0("Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=args.use_compile,
            checkpoint_path=args.checkpoint_path,
            load_from_HF=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=False,
        )

        print_rank0("Applying LoRA...")
        lora_config = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            apply_to_vision_encoder=args.apply_to_vision_encoder,
            apply_to_text_encoder=args.apply_to_text_encoder,
            apply_to_geometry_encoder=args.apply_to_geometry_encoder,
            apply_to_detr_encoder=args.apply_to_detr_encoder,
            apply_to_detr_decoder=args.apply_to_detr_decoder,
            apply_to_mask_decoder=args.apply_to_mask_decoder,
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        stats = count_parameters(self.model)
        print_rank0(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        self.model.to(self.device)

        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            print_rank0("Model wrapped with DistributedDataParallel")

        self._unwrapped_model = self.model.module if self.multi_gpu else self.model

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )

        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        loss_fns = [
            Boxes(weight_dict={
                "loss_bbox": 5.0,
                "loss_giou": 2.0
            }),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict={
                    "loss_ce": 20.0,
                    "presence_loss": 20.0
                },
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=200,
            ),
            Masks(
                weight_dict={
                    "loss_mask": 200.0,
                    "loss_dice": 10.0
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False
            )
        ]

        o2m_matcher = BinaryOneToManyMatcher(
            alpha=0.3,
            threshold=0.4,
            topk=4
        )

        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",
            normalize_by_valid_object_num=False,
        )

    def train(self):
        data_dir = self.args.data_dir

        print_rank0(f"\nLoading training data from {data_dir}...")
        train_ds = FolderSegmentDataset(
            data_dir=data_dir,
            csv_path=self.args.train_csv_path,
            split="train",
        )

        has_validation = False
        val_ds = None

        if self.args.val_csv_path is not None:
            try:
                print_rank0(f"\nLoading validation data from {self.args.val_csv_path}...")
                val_ds = FolderSegmentDataset(
                    data_dir=data_dir,
                    csv_path=self.args.val_csv_path,
                    split="val",
                )
                if len(val_ds) > 0:
                    has_validation = True
                    print_rank0(f"Found validation data: {len(val_ds)} images")
                else:
                    print_rank0("Validation dataset is empty.")
                    val_ds = None
            except Exception as e:
                print_rank0(f"Could not load validation data: {e}")
                val_ds = None

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        train_sampler = None
        val_sampler = None

        if self.multi_gpu:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.world_size,
                    rank=get_rank(),
                    shuffle=False
                )

        if not self.args.num_workers:
            train_num_workers = recommended_num_workers(train=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=train_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        if has_validation:
            if not self.args.num_workers:
                val_num_workers = recommended_num_workers(train=False)

            val_loader = DataLoader(
                val_ds,
                batch_size=self.args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=val_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            val_loader = None

        self.model.train()

        epochs = self.args.num_epochs
        best_val_loss = float("inf")
        print_rank0(f"Starting training for {epochs} epochs...")

        if has_validation:
            print_rank0(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("⚠️  No validation data found - training without validation")

        if self.multi_gpu:
            print_rank0(
                f"Effective batch size: {self.args.batch_size} x {self.world_size} = {self.args.batch_size * self.world_size}"
            )

        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_losses = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for batch_dict in pbar:
                input_batch = batch_dict["input"]
                input_batch = move_to_device(input_batch, self.device)

                outputs_list = self.model(input_batch)

                find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            outputs["indices"] = self.matcher(outputs, targets)
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                loss_dict = self.loss_wrapper(outputs_list, find_targets)
                total_loss = loss_dict[CORE_LOSS_KEY]

                self.optimizer.zero_grad()
                total_loss.backward()

                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.args.max_grad_norm
                    )

                self.optimizer.step()

                train_losses.append(total_loss.item())
                pbar.set_postfix({"loss": total_loss.item()})

            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

            if has_validation and val_loader is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc="Validation", disable=not is_main_process())

                    for batch_dict in val_pbar:
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)

                        outputs_list = self.model(input_batch)

                        find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        with SAM3Output.iteration_mode(
                            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                                stage_targets_list = [stage_targets] * len(stage_outputs)
                                for outputs, targets in zip(stage_outputs, stage_targets_list):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(aux_out, targets)

                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]

                        val_losses.append(total_loss.item())
                        val_pbar.set_postfix({"val_loss": total_loss.item()})

                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

                if self.multi_gpu:
                    val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                print_rank0(
                    f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )

                if is_main_process():
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_lora_weights(model_to_save, str(out_dir / "best_lora_weights.pt"))
                        print(f"✓ New best model saved (val_loss: {avg_val_loss:.6f})")

                    with open(out_dir / "val_stats.json", "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss
                        }) + "\n")

                torch.cuda.empty_cache()
                self.model.train()
            else:
                print_rank0(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
                if is_main_process():
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

        if self.multi_gpu:
            dist.barrier()

        if is_main_process():
            if has_validation:
                print(f"\n{'='*80}")
                print("✅ Training complete!")
                print(f"{'='*80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print("  - best_lora_weights.pt (best validation loss)")
                print("  - last_lora_weights.pt (last epoch)")
                print(f"\n📊 To compute full metrics (mAP, cgF1) with NMS:")
                print("   python validate_sam3_lora.py \\")
                print(f"     --weights {out_dir}/best_lora_weights.pt \\")
                print(f"     --val_data_dir {data_dir}/valid")
                print(f"{'='*80}")
            else:
                last_path = out_dir / "last_lora_weights.pt"
                best_path = out_dir / "best_lora_weights.pt"
                if last_path.exists():
                    shutil.copy(last_path, best_path)

                print(f"\n{'='*80}")
                print("✅ Training complete!")
                print(f"{'='*80}")
                print(f"\nModels saved to {out_dir}:")
                print("  - best_lora_weights.pt (copy of last epoch)")
                print("  - last_lora_weights.pt (last epoch)")
                print("\nℹ️  No validation data - consider adding data/valid/ for better model selection")
                print(f"{'='*80}")

        if self.multi_gpu:
            cleanup_distributed()


# ============================================================================
# Launch helper
# ============================================================================

def launch_distributed_training(args):
    """Launch training with multiple GPUs using torchrun subprocess."""
    import subprocess

    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching distributed training on GPUs: {devices}")
    print(f"Number of processes: {num_gpus}")

    forwarded_args = [a for a in sys.argv[1:] if a != "--_launched_by_torchrun"]

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port", str(args.master_port),
        sys.argv[0],
        *forwarded_args,
        "--_launched_by_torchrun",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA (CLI-only config)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU:
    python train_sam3_petct.py --data_dir /workspace/data --device 0

  Multi-GPU:
    python train_sam3_petct.py --data_dir /workspace/data --device 0 1
        """
    )

    # Runtime / distributed
    parser.add_argument("--device", type=int, nargs="+", default=[0],
                        help="GPU device ID(s) to use")
    parser.add_argument("--master_port", type=int, default=29500,
                        help="Master port for distributed training")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--_launched_by_torchrun", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "qkv", "proj", "fc1", "fc2",
            "c_fc", "c_proj",
            "linear1", "linear2",
        ],
        help="List of module names to apply LoRA to"
    )

    parser.add_argument("--apply_to_vision_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply_to_text_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply_to_geometry_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply_to_detr_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply_to_detr_decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply_to_mask_decoder", action=argparse.BooleanOptionalAction, default=True)

    # Training
    parser.add_argument("--data_dir", type=str, default="/workspace/data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--train_csv_path", type=str, required=True,
                    help="Path to training CSV file")
    parser.add_argument("--val_csv_path", type=str, default=None,
                        help="Path to validation CSV file")
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/sam3_lora_full")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--save_lora_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hub_model_id", type=str, default=None)

    # Evaluation / hardware
    parser.add_argument("--metric", type=str, default="iou")
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compute_metrics_during_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hardware_device", type=str, default="cuda")
    parser.add_argument("--dataloader_pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_compile", action=argparse.BooleanOptionalAction, default=False)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("Training configuration:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    num_devices = len(args.device)
    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ

    if num_devices > 1 and not is_torchrun_subprocess:
        launch_distributed_training(args)
    else:
        multi_gpu = num_devices > 1 and is_torchrun_subprocess

        if not multi_gpu and num_devices == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
            print(f"Using single GPU: {args.device[0]}")

        trainer = SAM3TrainerNative(args, multi_gpu=multi_gpu)
        trainer.train()


if __name__ == "__main__":
    # python train_sam3_petct.py \
    # --data_dir /workspace/data \
    # --output_dir outputs/sam3_lora_full \
    # --batch_size 4 \
    # --num_epochs 100 \
    # --learning_rate 5e-5 \
    # --weight_decay 0.01 \
    # --lora_rank 16 \
    # --lora_alpha 32 \
    # --lora_dropout 0.1 \
    # --device 0 1
    main()