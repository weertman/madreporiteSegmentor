# train.py
import argparse
import os
import csv
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataloader import (get_tasks_dict, visualize_image_pair_count_distribution, visualize_transforms,
                        random_split_tasks, get_pairs_from_tasks, SegmentationDataset)
# Existing metrics imported:
from metrics import DiceCoefficient, IoU
# ===============================================
# NEW IMPORT for the ProbMaxInMask metric:
from metrics import ProbMaxInMask
# ===============================================
from custom_transformations import EmphasisCrop
from plotting import visualize_reconstructions, plot_iteration_loss, plot_metrics_log
from resnet import PVTv2ResNetSegmentationModel
from unet import PVTv2UNetSegmentationModel

# Attempt to import wandb. If not installed, or no connection, handle gracefully.
try:
    import wandb
except ImportError:
    wandb = None


def train_one_epoch(model, dataloader, optimizer, device, criterion, epoch, iter_loss_writer, wandb_run):
    model.train()
    running_loss = 0.0

    # Existing metrics:
    dice = DiceCoefficient()
    iou = IoU()

    # ===============================================
    # NEW METRIC: ProbMaxInMask
    pmim = ProbMaxInMask()
    # ===============================================

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training Epoch {epoch+1}", leave=False)
    for batch_idx, (images, masks) in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks.unsqueeze(1).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        dice.update(logits, masks)
        iou.update(logits, masks)
        # ===============================================
        pmim.update(logits, masks)
        # ===============================================

        pbar.set_postfix(loss=loss.item())
        iter_loss_writer.writerow([epoch+1, 'train', batch_idx+1, loss.item()])

        # Log iteration-level metrics to W&B if available (no step specified)
        if wandb_run is not None:
            try:
                wandb.log({"train_iteration_loss": loss.item()})
            except:
                pass  # If logging fails, continue without crashing

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = dice.compute()
    epoch_iou = iou.compute()
    # ===============================================
    epoch_pmim = pmim.compute()
    # ===============================================
    return epoch_loss, epoch_dice, epoch_iou, epoch_pmim


def evaluate(model, dataloader, device, criterion, epoch, iter_loss_writer, wandb_run, phase="val"):
    model.eval()
    running_loss = 0.0

    dice = DiceCoefficient()
    iou = IoU()
    # ===============================================
    pmim = ProbMaxInMask()
    # ===============================================

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating Epoch {epoch+1}", leave=False)
    with torch.no_grad():
        for batch_idx, (images, masks) in pbar:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks.unsqueeze(1).float())

            running_loss += loss.item() * images.size(0)
            dice.update(logits, masks)
            iou.update(logits, masks)
            # ===============================================
            pmim.update(logits, masks)
            # ===============================================
            pbar.set_postfix(loss=loss.item())

            iter_loss_writer.writerow([epoch+1, phase, batch_idx+1, loss.item()])

            # Log iteration-level evaluation metrics if available (no step specified)
            if wandb_run is not None:
                try:
                    wandb.log({f"{phase}_iteration_loss": loss.item()})
                except:
                    pass

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = dice.compute()
    epoch_iou = iou.compute()
    # ===============================================
    epoch_pmim = pmim.compute()
    # ===============================================
    return epoch_loss, epoch_dice, epoch_iou, epoch_pmim


def evaluate_final(model, dataloader, device, criterion):
    """
    Evaluate the model on a given dataloader without logging iteration-level details.
    Returns loss, dice, iou, and pmim for the entire dataloader.
    """
    model.eval()
    running_loss = 0.0
    dice = DiceCoefficient()
    iou = IoU()
    # ===============================================
    pmim = ProbMaxInMask()
    # ===============================================
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks.unsqueeze(1).float())
            running_loss += loss.item() * images.size(0)
            dice.update(logits, masks)
            iou.update(logits, masks)
            # ===============================================
            pmim.update(logits, masks)
            # ===============================================

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = dice.compute()
    epoch_iou = iou.compute()
    # ===============================================
    epoch_pmim = pmim.compute()
    # ===============================================
    return epoch_loss, epoch_dice, epoch_iou, epoch_pmim


def compute_pos_weight(dataset, device, num_workers=4, max_pos_weight=1000, default_pos_weight=10.0):
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=num_workers)
    total_pixels = 0
    positive_pixels = 0

    print("Computing pos_weight by scanning dataset (this may take a while)...")
    for images, masks in tqdm(loader, desc="Counting pixels"):
        masks_flat = masks.view(-1)
        positive_count = (masks_flat == 1).sum().item()
        total_count = masks_flat.numel()
        positive_pixels += positive_count
        total_pixels += total_count

    if positive_pixels == 0:
        print("No positive pixels found! Falling back to default pos_weight.")
        return default_pos_weight

    negative_pixels = total_pixels - positive_pixels
    pos_weight = negative_pixels / positive_pixels
    if pos_weight > max_pos_weight:
        print(f"Computed pos_weight = {pos_weight:.2f} exceeds max_pos_weight={max_pos_weight}. Using max_pos_weight.")
        pos_weight = max_pos_weight
    else:
        print(f"Computed pos_weight = {pos_weight:.2f} (pos={positive_pixels}, neg={negative_pixels})")
    return pos_weight


def create_scheduler(optimizer, scheduler_config):
    scheduler_type = scheduler_config.get("type", None)
    if scheduler_type == 'reduce_on_plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.1),
            patience=scheduler_config.get("patience", 5)
        )
    elif scheduler_type == 'step_lr':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("gamma", 0.1)
        )
    elif scheduler_type == 'cosine_annealing':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 50),
            eta_min=scheduler_config.get("eta_min", 0)
        )
    elif scheduler_type is None:
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def build_transforms(config):
    tf_cfg = config["transforms"]
    input_height = config["data"]["input_shape"][1]
    input_width = config["data"]["input_shape"][0]

    transform_list = [
        EmphasisCrop(
            p=tf_cfg["EmphasisCrop"].get("p", 0.5),
            scale_factor=tf_cfg["EmphasisCrop"].get("scale_factor", 10.0),
            scale_factor_range=tf_cfg["EmphasisCrop"].get("scale_factor_range", (0.2, 1.3)),
            jitter_ratio=tf_cfg["EmphasisCrop"].get("jitter_ratio", 0.2),
            jitter_ratio_range=tf_cfg["EmphasisCrop"].get("jitter_ratio_range", (0.8, 1.2))
        ),
        A.ShiftScaleRotate(
            p=tf_cfg["ShiftScaleRotate"].get("p", 0.5),
            shift_limit=tf_cfg["ShiftScaleRotate"].get("shift_limit", 0.1),
            scale_limit=tf_cfg["ShiftScaleRotate"].get("scale_limit", 0.2),
            rotate_limit=tf_cfg["ShiftScaleRotate"].get("rotate_limit", 45)
        ),
        A.Resize(height=input_height, width=input_width),
        A.HorizontalFlip(p=tf_cfg["HorizontalFlip"].get("p", 0.5)),
        A.VerticalFlip(p=tf_cfg["VerticalFlip"].get("p", 0.5)),
        A.ToGray(p=tf_cfg["ToGray"].get("p", 0.2)),
        A.HueSaturationValue(
            p=tf_cfg["HueSaturationValue"].get("p", 0.2),
            hue_shift_limit=tf_cfg["HueSaturationValue"].get("hue_shift_limit", 20),
            sat_shift_limit=tf_cfg["HueSaturationValue"].get("sat_shift_limit", 30),
            val_shift_limit=tf_cfg["HueSaturationValue"].get("val_shift_limit", 20),
        ),
        A.RandomBrightnessContrast(
            p=tf_cfg["RandomBrightnessContrast"].get("p", 0.2),
            brightness_limit=tf_cfg["RandomBrightnessContrast"].get("brightness_limit", 0.2),
            contrast_limit=tf_cfg["RandomBrightnessContrast"].get("contrast_limit", 0.2)
        ),
        A.ColorJitter(
            p=tf_cfg["ColorJitter"].get("p", 0.5),
            brightness=tf_cfg["ColorJitter"].get("brightness", 0.2),
            contrast=tf_cfg["ColorJitter"].get("contrast", 0.2),
            saturation=tf_cfg["ColorJitter"].get("saturation", 0.2),
            hue=tf_cfg["ColorJitter"].get("hue", 0.2)
        ),
        A.Normalize(
            mean=tf_cfg["Normalize"].get("mean", (0.485, 0.456, 0.406)),
            std=tf_cfg["Normalize"].get("std", (0.229, 0.224, 0.225))
        ),
        ToTensorV2()
    ]

    transform = A.Compose(transform_list, additional_targets={'mask': 'mask'})
    return transform


def build_eval_transforms(config):
    tf_cfg = config["transforms"]
    input_height = config["data"]["input_shape"][1]
    input_width = config["data"]["input_shape"][0]

    eval_transform_list = [
        A.Resize(height=input_height, width=input_width),
        A.Normalize(
            mean=tf_cfg["Normalize"].get("mean", (0.485, 0.456, 0.406)),
            std=tf_cfg["Normalize"].get("std", (0.229, 0.224, 0.225))
        ),
        ToTensorV2()
    ]

    eval_transform = A.Compose(eval_transform_list, additional_targets={'mask': 'mask'})
    return eval_transform


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config["paths"]["dataset"])
    run_dir = Path(config["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(run_dir / "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_shape = config["data"]["input_shape"]

    # Try to initialize W&B logging
    wandb_run = None
    if wandb is not None:
        try:
            wandb_run = wandb.init(project=config["training"]["wandb_project"], name=run_dir.name, config=config)
        except:
            print("Warning: Failed to connect to Weights & Biases. Continuing without W&B logging.")
            wandb_run = None

    # Get tasks
    tasks_dict = get_tasks_dict(dataset_path)
    print(f'Number of tasks: {len(tasks_dict)}')

    # Visualize distribution
    visualize_image_pair_count_distribution(run_dir, tasks_dict)

    # Split tasks
    random_seed = config["data"].get("random_seed", None)
    train_tasks, test_tasks, val_tasks = random_split_tasks(
        tasks_dict,
        config["data"]["train_ratio"],
        config["data"]["test_ratio"],
        config["data"]["val_ratio"],
        random_seed=random_seed
    )
    print(f"Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}, Val tasks: {len(val_tasks)}")

    train_pairs = get_pairs_from_tasks(tasks_dict, train_tasks)
    test_pairs = get_pairs_from_tasks(tasks_dict, test_tasks)
    val_pairs = get_pairs_from_tasks(tasks_dict, val_tasks)

    # Build transforms
    train_transform = build_transforms(config)
    eval_transform = build_eval_transforms(config)

    # Visualize transforms (training transforms)
    for i in range(0, 6):
        visualize_transforms(run_dir, train_pairs, train_transform, name=f'transform_visualization_{i}.png')

    # Datasets and loaders
    train_dataset = SegmentationDataset(train_pairs, transform=train_transform)
    test_dataset = SegmentationDataset(test_pairs, transform=eval_transform)
    val_dataset = SegmentationDataset(val_pairs, transform=eval_transform)

    batch_size = config["dataloader"]["batch_size"]
    num_workers = config["dataloader"]["num_workers"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model parameters
    decoder_type = config["model"]["decoder_type"]
    backbone = config["model"]["backbone"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"]["pretrained"]
    freeze_backbone = config["model"]["freeze_backbone"]
    decoder_init_method = config["model"]["decoder_init_method"]
    decoder_blocks = config["model"]["decoder_blocks"]

    # Training parameters
    num_epochs = config["training"]["num_epochs"]
    freeze_backbone_epochs = config["training"]["freeze_backbone_epochs"]
    lr = config["training"]["lr"]
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else 'cpu')
    USE_AUTO_POS_WEIGHT = config["training"]["USE_AUTO_POS_WEIGHT"]
    DEFAULT_POS_WEIGHT = config["training"]["DEFAULT_POS_WEIGHT"]
    max_pos_weight = config["training"]["max_pos_weight"]

    # Build model
    if decoder_type == 'resnet':
        model = PVTv2ResNetSegmentationModel(
            model_name=f"OpenGVLab/{backbone}",
            num_classes=num_classes,
            pretrained=pretrained,
            decoder_blocks=decoder_blocks,
            freeze_backbone=freeze_backbone,
            decoder_init_method=decoder_init_method
        ).to(device)
    elif decoder_type == 'unet':
        model = PVTv2UNetSegmentationModel(
            model_name=f"OpenGVLab/{backbone}",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            decoder_init_method=decoder_init_method,
            input_shape=input_shape
        ).to(device)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    # Compute pos_weight if needed
    if USE_AUTO_POS_WEIGHT:
        computed_pos_weight = compute_pos_weight(train_dataset, device=device, num_workers=num_workers,
                                                 max_pos_weight=max_pos_weight, default_pos_weight=DEFAULT_POS_WEIGHT)
        pos_weight = torch.tensor([computed_pos_weight], device=device)
    else:
        pos_weight = torch.tensor([DEFAULT_POS_WEIGHT], device=device)

    # Optimizer
    optimizer_type = config["optimizer"]["type"]
    opt_lr = config["optimizer"]["lr"]

    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Criterion
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Scheduler
    scheduler_config = config.get("scheduler", {})
    scheduler = create_scheduler(optimizer, scheduler_config)
    scheduler_type = scheduler_config.get("type", None)

    best_val_loss = float('inf')
    metrics_file = run_dir / "metrics_log.csv"
    iteration_loss_file = run_dir / "iteration_loss.csv"

    if metrics_file.exists():
        os.remove(metrics_file)
    if iteration_loss_file.exists():
        os.remove(iteration_loss_file)

    write_header = not metrics_file.exists()
    iter_write_header = not iteration_loss_file.exists()

    num_samples = 5
    recons_dir = run_dir / 'recons'
    recons_dir.mkdir(parents=True, exist_ok=True)

    with open(iteration_loss_file, 'a', newline='') as iter_f:
        iter_writer = csv.writer(iter_f)
        if iter_write_header:
            iter_writer.writerow(["epoch", "phase", "iteration", "loss"])

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Unfreeze backbone if needed
            if epoch == freeze_backbone_epochs:
                print("Unfreezing backbone...")
                for param in model.backbone.parameters():
                    param.requires_grad = True

                # Re-create optimizer now that backbone is trainable
                if optimizer_type == 'adamw':
                    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr)
                elif optimizer_type == 'sgd':
                    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr)
                elif optimizer_type == 'adam':
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr)

                scheduler = create_scheduler(optimizer, scheduler_config)

            # ===============================================
            # TRAIN + EVAL now return pmim as well
            train_loss, train_dice, train_iou, train_pmim = train_one_epoch(
                model, train_loader, optimizer, device, criterion, epoch, iter_writer, wandb_run
            )
            val_loss, val_dice, val_iou, val_pmim = evaluate(
                model, val_loader, device, criterion, epoch, iter_writer, wandb_run, phase="val"
            )
            # ===============================================

            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}, Train PMIM: {train_pmim:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val PMIM: {val_pmim:.4f}")

            # Log epoch-level metrics to W&B if available (no step specified)
            if wandb_run is not None:
                try:
                    wandb.log({
                        "epoch": epoch+1,
                        "train_loss": train_loss,
                        "train_dice": train_dice,
                        "train_iou": train_iou,
                        "train_pmim": train_pmim,  # NEW
                        "val_loss": val_loss,
                        "val_dice": val_dice,
                        "val_iou": val_iou,
                        "val_pmim": val_pmim       # NEW
                    })
                except:
                    pass

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / 'best_model.pth')
                print("Best model saved.")

            torch.save(model.state_dict(), output_dir / 'current_model.pth')

            with open(metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "epoch",
                        "train_loss", "train_dice", "train_iou", "train_pmim",
                        "val_loss",   "val_dice",   "val_iou",   "val_pmim"
                    ])
                    write_header = False
                writer.writerow([
                    epoch+1,
                    train_loss, train_dice, train_iou, train_pmim,
                    val_loss,   val_dice,   val_iou,   val_pmim
                ])

            # Visualization samples (from test_loader, which uses eval_transform)
            if len(test_dataset) == 0:
                print("No test samples found. Skipping reconstruction visualization.")
            else:
                # Choose random indices
                random_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))

                images_list = []
                masks_list = []
                for idx in random_indices:
                    img, mask = test_dataset[idx]  # calls Dataset.__getitem__(idx)
                    images_list.append(img)
                    masks_list.append(mask)

                # Stack them into tensors of shape (N, C, H, W) and (N, H, W)
                images_tensor = torch.stack(images_list, dim=0)
                masks_tensor = torch.stack(masks_list, dim=0)

                # Now visualize
                path_fig = visualize_reconstructions(
                    model,
                    device,
                    list(zip(images_tensor, masks_tensor)),
                    epoch,
                    recons_dir
                )

            if epoch > 0:
                plot_metrics_log(run_dir)
                iter_f.flush()
                plot_iteration_loss(run_dir)

            # Update scheduler if needed
            if scheduler is not None and scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_loss)
            elif scheduler is not None and scheduler_type != 'reduce_on_plateau':
                scheduler.step()

    print("Training complete.")

    # Final evaluation on train, val, test sets using best model
    best_model_path = output_dir / 'best_model.pth'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        print("Loaded best model for final evaluation.")

        # ===============================================
        # Evaluate final also returns pmim
        final_train_loss, final_train_dice, final_train_iou, final_train_pmim = evaluate_final(model, train_loader, device, criterion)
        final_val_loss, final_val_dice, final_val_iou, final_val_pmim = evaluate_final(model, val_loader, device, criterion)
        final_test_loss, final_test_dice, final_test_iou, final_test_pmim = evaluate_final(model, test_loader, device, criterion)
        # ===============================================

        # Save results to results.csv
        results_file = run_dir / "results.csv"
        with open(results_file, 'w', newline='') as rf:
            writer = csv.writer(rf)
            # ===============================================
            writer.writerow(["split", "loss", "dice", "iou", "pmim"])
            # ===============================================
            writer.writerow(["train", final_train_loss, final_train_dice, final_train_iou, final_train_pmim])
            writer.writerow(["val",   final_val_loss,   final_val_dice,   final_val_iou,   final_val_pmim])
            writer.writerow(["test",  final_test_loss,  final_test_dice,  final_test_iou,  final_test_pmim])

        print("Final evaluation results saved to results.csv.")

        # Log final results to W&B if available (no step specified)
        if wandb_run is not None:
            try:
                wandb.log({
                    "final_train_loss": final_train_loss,
                    "final_train_dice": final_train_dice,
                    "final_train_iou": final_train_iou,
                    "final_train_pmim": final_train_pmim,  # NEW
                    "final_val_loss": final_val_loss,
                    "final_val_dice": final_val_dice,
                    "final_val_iou": final_val_iou,
                    "final_val_pmim": final_val_pmim,       # NEW
                    "final_test_loss": final_test_loss,
                    "final_test_dice": final_test_dice,
                    "final_test_iou": final_test_iou,
                    "final_test_pmim": final_test_pmim      # NEW
                })
            except:
                pass
    else:
        print("No best model found for final evaluation.")

    # Finish W&B run if active
    if wandb_run is not None:
        try:
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model using a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    main(args.config)
