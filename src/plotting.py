# plotting.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch

def visualize_reconstructions(model, device, samples, epoch, recons_dir, confidence_threshold=0.5):
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    n = len(samples)

    fig, axs = plt.subplots(n, 3, figsize=(12, 4*n))

    if n == 1:
        axs = [axs]

    with torch.no_grad():
        for i, (img, mask) in enumerate(samples):
            img_in = img.unsqueeze(0).to(device)
            logits = model(img_in)
            pred = torch.sigmoid(logits).cpu().squeeze(0).squeeze(0).numpy()

            img_np = img.permute(1,2,0).cpu().numpy()
            img_np = (img_np * std) + mean
            img_np = np.clip(img_np, 0, 1)

            mask_np = mask.cpu().numpy()

            axs[i][0].imshow(img_np)
            axs[i][0].set_title("Input")
            axs[i][0].axis('off')

            axs[i][1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axs[i][1].set_title("Ground Truth")
            axs[i][1].axis('off')

            axs[i][2].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axs[i][2].set_title("Prediction")
            axs[i][2].axis('off')

    fig.tight_layout()
    path_fig = recons_dir / f'{epoch+1}_recon.png'
    plt.savefig(path_fig, dpi=150)
    plt.close(fig)
    print(f"Saved reconstruction visualization to {path_fig}")

def plot_metrics_log(run_dir: Path):
    metrics_file = run_dir / "metrics_log.csv"
    if not metrics_file.exists():
        print(f"File {metrics_file} not found. Cannot plot metrics.")
        return

    df = pd.read_csv(metrics_file)

    # Create a figure with three subplots: Loss, Dice, IoU
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # Plot Loss
    axs[0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axs[0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o')
    axs[0].set_title('Loss Over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Dice
    axs[1].plot(df['epoch'], df['train_dice'], label='Train Dice', marker='o')
    axs[1].plot(df['epoch'], df['val_dice'], label='Val Dice', marker='o')
    axs[1].set_title('Dice Over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Dice')
    axs[1].legend()
    axs[1].grid(True)

    # Plot IoU
    axs[2].plot(df['epoch'], df['train_iou'], label='Train IoU', marker='o')
    axs[2].plot(df['epoch'], df['val_iou'], label='Val IoU', marker='o')
    axs[2].set_title('IoU Over Epochs')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('IoU')
    axs[2].legend()
    axs[2].grid(True)

    # Plot PMIM
    axs[3].plot(df['epoch'], df['train_pmim'], label='Train PMIM', marker='o')
    axs[3].plot(df['epoch'], df['val_pmim'], label='Val PMIM', marker='o')
    axs[3].set_title('PMIM Over Epochs')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('PMIM')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    save_path = run_dir / 'metrics_log.png'
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Metrics log plot saved to {save_path}")


def plot_iteration_loss(run_dir: Path):
    iteration_loss_file = run_dir / "iteration_loss.csv"
    if not iteration_loss_file.exists():
        print(f"File {iteration_loss_file} not found. Cannot plot iteration loss.")
        return

    df = pd.read_csv(iteration_loss_file)

    # Separate phases (train, val) for clarity
    train_df = df[df['phase'] == 'train']
    val_df = df[df['phase'] == 'val']

    # Sort by epoch and iteration for consistency
    train_df = train_df.sort_values(by=['epoch', 'iteration'])
    val_df = val_df.sort_values(by=['epoch', 'iteration'])

    # Create a global iteration counter for train and val independently
    # This shows progression through training/validation sets across all epochs
    max_train_iter = train_df['iteration'].max() if not train_df.empty else 0
    max_val_iter = val_df['iteration'].max() if not val_df.empty else 0

    train_df['global_iter'] = (train_df['epoch'] - 1) * max_train_iter + train_df['iteration']
    val_df['global_iter'] = (val_df['epoch'] - 1) * max_val_iter + val_df['iteration']

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training iteration loss
    if not train_df.empty:
        axs[0].plot(train_df['global_iter'], train_df['loss'], label='Train Loss', alpha=0.7, color='blue')
        axs[0].set_title('Training Iteration-Level Loss')
        axs[0].set_xlabel('Global Training Iteration')
        axs[0].set_ylabel('Loss')
        axs[0].grid(True)
        axs[0].legend()
    else:
        axs[0].text(0.5, 0.5, 'No Training Data', ha='center', va='center', transform=axs[0].transAxes)

    # Plot validation iteration loss
    if not val_df.empty:
        axs[1].plot(val_df['global_iter'], val_df['loss'], label='Val Loss', alpha=0.7, color='orange')
        axs[1].set_title('Validation Iteration-Level Loss')
        axs[1].set_xlabel('Global Validation Iteration')
        axs[1].set_ylabel('Loss')
        axs[1].grid(True)
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=axs[1].transAxes)

    plt.tight_layout()
    save_path = run_dir / 'iteration_loss.png'
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Iteration loss plot saved to {save_path}")
