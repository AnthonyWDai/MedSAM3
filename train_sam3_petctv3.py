"""
SAM3 LoRA Training Script (folder-based segmentation dataset, simple queries)

Dataset structure:
  data_dir/
    train/
      images/
        patient_or_class_dir/.../case.png
      masks/
        patient_or_class_dir/.../case.png
    val/
      images/
        patient_or_class_dir/.../case.png
      masks/
        patient_or_class_dir/.../case.png

Notes:
  - No CSV is used
  - Queries are built directly from mask-derived objects
  - All objects are assigned class name "lesion"
  - If no annotations exist, a single fallback query "no lesion" is created
"""

import os
import sys
import json
import argparse
import random
import subprocess
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.transforms import v2

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image,
    Object,
    FindQueryLoaded,
    InferenceMetadata,
)

from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_rank0(*args, **kwargs):
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


def launch_distributed_training(args):
    num_gpus = len(args.device)
    device_str = ",".join(map(str, args.device))

    forwarded_args = []
    skip_next = False
    for token in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--device":
            skip_next = True
            continue
        if token in map(str, args.device):
            continue
        forwarded_args.append(token)

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
# Metrics
# ============================================================================

def compute_seg_dice_stats(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 1, eps: float = 1e-6):
    """
    pred, target: [B, H, W] integer tensors
    For binary segmentation, num_classes=1 assumes foreground class == 1.
    """
    dice_sum = 0.0
    valid_count = 0

    reduce_dims = tuple(range(1, pred.ndim))

    for cls in range(1, num_classes + 1):
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()

        intersect = (pred_c * target_c).sum(dim=reduce_dims)
        pred_sum = pred_c.sum(dim=reduce_dims)
        target_sum = target_c.sum(dim=reduce_dims)
        denom = pred_sum + target_sum
        valid = denom > 0

        if valid.any():
            dice = (2.0 * intersect[valid] + eps) / (denom[valid] + eps)
            dice_sum += dice.sum().item()
            valid_count += valid.sum().item()

    return dice_sum, valid_count


# ============================================================================
# Dataset
# ============================================================================

class FolderSegmentDataset(Dataset):
    """
    Folder-based segmentation dataset.

    Expected structure:
      data_dir/
        split/
          images/
            **/*.{png,jpg,jpeg,bmp,tif,tiff}
          masks/
            same relative paths / filenames as images

    Notes:
      - No CSV reading
      - All extracted objects get class name "lesion"
      - Queries are built from unique class names mapped to object IDs
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        data_dir,
        split="train",
        resolution=1008,
        min_instance_area=100,
        default_label="lesion",
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.resolution = resolution
        self.min_instance_area = min_instance_area
        self.default_label = str(default_label).strip().lower() or "lesion"

        self.split_dir = self.data_dir / split
        self.images_dir = self.split_dir / "images"
        self.masks_dir = self.split_dir / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.samples = []
        sample_id = 0

        image_files = sorted([
            p for p in self.images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMG_EXTS
        ])

        for img_path in image_files:
            rel_path = img_path.relative_to(self.images_dir)
            mask_path = self.masks_dir / rel_path

            if not mask_path.exists():
                print(f"Warning: no mask found for image {img_path}, expected {mask_path}")
                continue

            self.samples.append({
                "id": sample_id,
                "image_path": img_path,
                "mask_path": mask_path,
            })
            sample_id += 1

        print(f"Loaded folder dataset: {split} split")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Default label: {self.default_label}")

    def __len__(self):
        return len(self.samples)

    def _binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.max() > 1:
            return (mask > 127).astype(np.uint8)
        return (mask > 0).astype(np.uint8)

    def extract_instance_masks(
        self,
        mask: np.ndarray,
        min_area: int = 100
    ) -> List[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        """
        Split a semantic mask into separate connected-instance masks.

        Returns:
            List of:
              (instance_mask, (x_center, y_center, width, height))
            where bbox is normalized to [0, 1] in original image coordinates
        """
        mask = self._binarize_mask(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = mask.shape
        instances = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < min_area:
                continue

            inst_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(inst_mask, [contour], contourIdx=-1, color=1, thickness=-1)

            x_center = (x + w / 2.0) / width
            y_center = (y + h / 2.0) / height
            width_norm = w / width
            height_norm = h / height

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width_norm = max(0.0, min(1.0, width_norm))
            height_norm = max(0.0, min(1.0, height_norm))

            instances.append((
                inst_mask,
                (x_center, y_center, width_norm, height_norm)
            ))

        return instances

    def build_simple_queries(
        self,
        objects,
        object_class_names,
        img_id,
        orig_h,
        orig_w,
    ):
        # Construct Queries - one per unique category
        # Each query maps to only the objects of that category
        class_to_object_ids = defaultdict(list)
        for obj, class_name in zip(objects, object_class_names):
            class_to_object_ids[str(class_name).lower()].append(obj.object_id)

        queries = []
        if len(class_to_object_ids) > 0:
            for order, (query_text, obj_ids) in enumerate(class_to_object_ids.items()):
                query = FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=order,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=0,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1
                    )
                )
                queries.append(query)
        else:
            query = FindQueryLoaded(
                query_text=f"no {self.default_label}",
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
            queries.append(query)

        return queries

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample["id"]
        img_path = sample["image_path"]
        mask_path = sample["mask_path"]

        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        resized_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)
        image_tensor = self.transform(resized_image)

        mask_image = PILImage.open(mask_path).convert("L")
        mask_np = np.array(mask_image)
        mask_bin = self._binarize_mask(mask_np)

        objects = []
        object_class_names = []

        if mask_bin.sum() > 0:
            instances = self.extract_instance_masks(
                mask_np,
                min_area=self.min_instance_area
            )

            for obj_id, (inst_mask, bbox_norm) in enumerate(instances):
                cx, cy, w, h = bbox_norm

                box_tensor = torch.tensor([cx, cy, w, h], dtype=torch.float32)

                mask_t = torch.from_numpy(inst_mask).float().unsqueeze(0).unsqueeze(0)
                mask_t = torch.nn.functional.interpolate(
                    mask_t,
                    size=(self.resolution, self.resolution),
                    mode="nearest"
                )
                segment = mask_t.squeeze(0).squeeze(0) > 0.5

                obj = Object(
                    bbox=box_tensor,
                    area=(box_tensor[2] * box_tensor[3]).item(),
                    object_id=obj_id,
                    segment=segment
                )
                objects.append(obj)
                object_class_names.append(self.default_label)

        queries = self.build_simple_queries(
            objects=objects,
            object_class_names=object_class_names,
            img_id=img_id,
            orig_h=orig_h,
            orig_w=orig_w,
        )

        image = Image(
            image=image_tensor,
            image_id=img_id,
            original_size=(orig_h, orig_w),
        )

        datapoint = Datapoint(
            image=image,
            objects=objects,
            find_queries=queries,
        )

        return {"input": datapoint}


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
            split="train",
            resolution=self.args.resolution,
            min_instance_area=self.args.min_instance_area,
            default_label=self.args.default_label,
        )

        has_validation = False
        val_ds = None

        val_dir = Path(data_dir) / "val"
        if val_dir.exists():
            try:
                print_rank0(f"\nLoading validation data from {val_dir}...")
                val_ds = FolderSegmentDataset(
                    data_dir=data_dir,
                    split="val",
                    resolution=self.args.resolution,
                    min_instance_area=self.args.min_instance_area,
                    default_label=self.args.default_label,
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

        train_num_workers = self.args.num_workers if self.args.num_workers is not None else recommended_num_workers(train=True)

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
            val_num_workers = self.args.num_workers if self.args.num_workers is not None else recommended_num_workers(train=False)

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
        val_interval = max(1, epochs // 10)

        print_rank0(f"Starting training for {epochs} epochs...")
        print_rank0(f"Validation interval: every {val_interval} epoch(s) (~10% of total epochs)")

        if has_validation:
            print_rank0(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("No validation data found - training without validation")

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

            should_validate = has_validation and val_loader is not None and (
                ((epoch + 1) % val_interval == 0) or ((epoch + 1) == epochs)
            )

            if should_validate:
                self.model.eval()
                val_losses = []
                val_dice_sum = 0.0
                val_dice_count = 0

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

                        try:
                            final_outputs = outputs_list[-1] if isinstance(outputs_list, (list, tuple)) else outputs_list
                            if isinstance(final_outputs, list):
                                final_outputs = final_outputs[-1]

                            pred_masks = final_outputs.get("pred_masks", None)
                            pred_logits = final_outputs.get("pred_logits", None)

                            if pred_masks is not None and pred_logits is not None and pred_masks.ndim == 4:
                                pred_scores = torch.sigmoid(pred_logits.squeeze(-1))
                                best_q = pred_scores.argmax(dim=1)

                                batch_pred = []
                                batch_target = []

                                for b_idx, targets in enumerate(find_targets):
                                    pred_mask_b = pred_masks[b_idx, best_q[b_idx]]
                                    pred_bin = (torch.sigmoid(pred_mask_b) > 0.5).long()

                                    gt_mask = None

                                    if "masks" in targets and targets["masks"] is not None and len(targets["masks"]) > 0:
                                        gt_mask = targets["masks"][0]
                                        if gt_mask.ndim == 3:
                                            gt_mask = gt_mask[0]
                                        gt_mask = (gt_mask > 0.5).long()
                                    elif "segment" in targets and targets["segment"] is not None:
                                        gt_mask = targets["segment"]
                                        if gt_mask.ndim == 3:
                                            gt_mask = gt_mask[0]
                                        gt_mask = (gt_mask > 0.5).long()
                                    else:
                                        gt_mask = torch.zeros_like(pred_bin, dtype=torch.long)

                                    if gt_mask.shape != pred_bin.shape:
                                        gt_mask = torch.nn.functional.interpolate(
                                            gt_mask.float().unsqueeze(0).unsqueeze(0),
                                            size=pred_bin.shape[-2:],
                                            mode="nearest"
                                        ).squeeze(0).squeeze(0).long()

                                    batch_pred.append(pred_bin)
                                    batch_target.append(gt_mask)

                                if len(batch_pred) > 0:
                                    batch_pred = torch.stack(batch_pred, dim=0)
                                    batch_target = torch.stack(batch_target, dim=0)

                                    dice_sum, dice_count = compute_seg_dice_stats(
                                        batch_pred, batch_target, num_classes=1
                                    )
                                    val_dice_sum += dice_sum
                                    val_dice_count += dice_count

                        except Exception as e:
                            if is_main_process():
                                print(f"Warning: Dice computation failed on validation batch: {e}")

                        val_pbar.set_postfix({
                            "val_loss": total_loss.item(),
                            "val_dice": (val_dice_sum / val_dice_count) if val_dice_count > 0 else 0.0
                        })

                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
                avg_val_dice = val_dice_sum / val_dice_count if val_dice_count > 0 else 0.0

                if self.multi_gpu:
                    val_stats_tensor = torch.tensor(
                        [sum(val_losses), len(val_losses), val_dice_sum, val_dice_count],
                        device=self.device,
                        dtype=torch.float64
                    )
                    dist.all_reduce(val_stats_tensor, op=dist.ReduceOp.SUM)

                    total_val_loss_sum = val_stats_tensor[0].item()
                    total_val_loss_count = max(1.0, val_stats_tensor[1].item())
                    total_val_dice_sum = val_stats_tensor[2].item()
                    total_val_dice_count = val_stats_tensor[3].item()

                    avg_val_loss = total_val_loss_sum / total_val_loss_count
                    avg_val_dice = (
                        total_val_dice_sum / total_val_dice_count
                        if total_val_dice_count > 0 else 0.0
                    )

                print_rank0(
                    f"[Epoch {epoch+1}/{epochs}] "
                    f"train_loss={avg_train_loss:.6f} "
                    f"val_loss={avg_val_loss:.6f} "
                    f"val_dice={avg_val_dice:.6f}"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print_rank0(f"New best validation loss: {best_val_loss:.6f}")

                    if is_main_process():
                        ckpt_dir = out_dir / "best_checkpoint"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)

                        if self.args.save_lora_only:
                            save_lora_weights(self._unwrapped_model, ckpt_dir)
                        else:
                            torch.save(self._unwrapped_model.state_dict(), ckpt_dir / "pytorch_model.bin")

                        with open(ckpt_dir / "training_state.json", "w") as f:
                            json.dump({
                                "epoch": epoch + 1,
                                "best_val_loss": best_val_loss,
                                "avg_val_dice": avg_val_dice,
                                "default_label": self.args.default_label,
                            }, f, indent=2)

                self.model.train()
            else:
                print_rank0(f"[Epoch {epoch+1}/{epochs}] train_loss={avg_train_loss:.6f}")

        if is_main_process():
            final_dir = out_dir / "final_checkpoint"
            final_dir.mkdir(parents=True, exist_ok=True)

            if self.args.save_lora_only:
                save_lora_weights(self._unwrapped_model, final_dir)
            else:
                torch.save(self._unwrapped_model.state_dict(), final_dir / "pytorch_model.bin")

            with open(final_dir / "training_config.json", "w") as f:
                json.dump(vars(self.args), f, indent=2)

        if self.multi_gpu:
            cleanup_distributed()


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA using folder-based segmentation data and simple text queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Runtime / distributed
    parser.add_argument("--device", type=int, nargs="+", default=[0], help="GPU device ID(s) to use")
    parser.add_argument("--master_port", type=int, default=29500, help="Master port for distributed training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--_launched_by_torchrun", action="store_true", help=argparse.SUPPRESS)
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

    parser.add_argument(
        "--min_instance_area",
        type=int,
        default=100,
        help="Minimum area threshold for filtering small mask instances when converting semantic masks to connected instances"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input image and mask resize resolution"
    )
    parser.add_argument(
        "--default_label",
        type=str,
        default="lesion",
        help="Class/query label assigned to all extracted objects"
    )

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
    main()