# dataloader.py
from pathlib import Path
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_dataset_tasks(path_dataset):
    return [task for task in path_dataset.iterdir() if task.is_dir()]


def get_image_mask_pairs_in_task(task: Path) -> List[Tuple[Path, Path]]:
    image_dir = task / 'images'
    mask_dir = task / 'masks'
    image_paths = [image_dir / image.name for image in image_dir.iterdir() if image.is_file()]
    mask_paths = [mask_dir / mask.name for mask in mask_dir.iterdir() if mask.is_file()]

    # Make sure lists are aligned and sorted
    image_paths.sort()
    mask_paths.sort()

    # Check alignment by filename stem
    for img_p, mask_p in zip(image_paths, mask_paths):
        if img_p.stem != mask_p.stem:
            raise ValueError(f"Image {img_p.name} does not match Mask {mask_p.name}")

    return list(zip(image_paths, mask_paths))


def get_tasks_dict(path_dataset):
    tasks_dict = {}
    for task in get_dataset_tasks(path_dataset):
        pairs = get_image_mask_pairs_in_task(task)
        if pairs:
            tasks_dict[task.name] = pairs
    return tasks_dict


def visualize_image_pair_count_distribution(path_run_dir, tasks_dict):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    task_names = list(tasks_dict.keys())
    image_count = [len(tasks_dict[task]) for task in task_names]
    ax.hist(image_count, bins=range(0, max(image_count) + 1), alpha=0.75)
    ax.set_xlabel('Number of image-mask pairs')
    ax.set_ylabel('Number of tasks')
    ax.set_title('Image-mask pair count distribution')
    fig.savefig(path_run_dir / 'image_mask_pair_count_distribution.png')
    plt.close()


def random_split_tasks(tasks_dict, train_ratio, test_ratio, val_ratio, random_seed: Optional[int] = None):
    """
    Split tasks into train, test, val sets.

    If random_seed is provided, it fixes the seed for reproducible splits.
    Otherwise, splits differ each run.
    """
    if abs((train_ratio + test_ratio + val_ratio) - 1.0) > 1e-9:
        raise ValueError('Train, test, val ratios should sum to 1.0')

    task_names = list(tasks_dict.keys())

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    n_tasks = len(task_names)
    n_train = int(n_tasks * train_ratio)
    n_test = int(n_tasks * test_ratio)
    # Remaining go to val
    n_val = n_tasks - n_train - n_test

    random.shuffle(task_names)
    train_tasks = task_names[:n_train]
    test_tasks = task_names[n_train:n_train + n_test]
    val_tasks = task_names[n_train + n_test:]

    return train_tasks, test_tasks, val_tasks


class SegmentationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load image and mask using Pillow
        img = np.array(Image.open(img_path).convert("RGB"))  # H, W, 3
        mask = np.array(Image.open(mask_path).convert("L"))  # H, W
        mask = (mask > 127).astype(np.float32)  # Convert to {0,1}

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']  # C,H,W tensor
            mask = augmented['mask']  # H,W tensor (single channel)

        # Ensure mask is float32
        mask = mask.float()

        return img, mask


def get_pairs_from_tasks(tasks_dict, task_list):
    pairs = []
    for t in task_list:
        pairs.extend(tasks_dict[t])
    return pairs


def visualize_transforms(path_run_dir, pairs, transform, name='transform_visualization.png', n=5):
    """
    Visualizes the effect of transforms by applying them to a random image and mask n times.
    """
    img_path, mask_path = random.choice(pairs)

    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axs = plt.subplots(2, n+1, figsize=(4*(n+1), 8))
    fig.suptitle("Visualizing Transforms")

    # Show original
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    axs[1, 0].imshow(mask, cmap='gray')
    axs[1, 0].set_title("Original Mask")
    axs[1, 0].axis("off")

    # Apply transform n times
    for i in range(n):
        augmented = transform(image=img, mask=mask)
        img_t = augmented['image'].permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
        mask_t = augmented['mask'].cpu().numpy()

        # Denormalize
        img_hwc = (img_t * std) + mean
        img_hwc = np.clip(img_hwc, 0, 1)

        g = axs[0, i + 1].imshow(img_hwc)
        axs[0, i+1].set_title(f"Transformed Image {i+1}")
        axs[0, i+1].axis("off")

        g = axs[1, i+1].imshow(mask_t, cmap='gray')
        axs[1, i+1].set_title(f"Transformed Mask {i+1}")
        axs[1, i+1].axis("off")

    plt.tight_layout()
    save_path = path_run_dir / name
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Transform visualization saved to {save_path}")


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    # You can also pass this via command line if you prefer
    # For now, just hardcode or adjust as needed
    dataset_path = Path('../dataset/12-15-2024_madreporiteSegmentor')

    # Example ratios, adjust as needed
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    random_seed = 42

    print("Loading tasks...")
    tasks_dict = get_tasks_dict(dataset_path)
    print(f"Found {len(tasks_dict)} tasks.")

    # Split tasks
    train_tasks, test_tasks, val_tasks = random_split_tasks(
        tasks_dict,
        train_ratio,
        test_ratio,
        val_ratio,
        random_seed=random_seed
    )
    print(f"Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}, Val tasks: {len(val_tasks)}")

    train_pairs = get_pairs_from_tasks(tasks_dict, train_tasks)
    val_pairs = get_pairs_from_tasks(tasks_dict, val_tasks)
    test_pairs = get_pairs_from_tasks(tasks_dict, test_tasks)

    # Optionally define a simple transform (or leave it None)
    # Just as an example, no augmentation here, just normalization and to tensor
    # If you have albumentations installed, you can use the following lines:
    train_transform = A.Compose([
        A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.2, rotate_limit=45),
        A.Resize(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ToGray(p=0.2),
        A.HueSaturationValue(p=0.2, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        A.RandomBrightnessContrast(p=0.2, brightness_limit=0.2, contrast_limit=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
    eval_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    train_dataset = SegmentationDataset(train_pairs, transform=train_transform)
    val_dataset = SegmentationDataset(val_pairs, transform=eval_transform)
    test_dataset = SegmentationDataset(test_pairs, transform=eval_transform)

    batch_size = 8
    num_workers = 0  # set to 0 for debugging purposes, can raise if needed

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # Debug: fetch a batch from each loader and inspect
    def inspect_batch(dataloader, name="", iterations=5):
        print(f"\nInspecting {name} loader batch:")
        for i, (images, masks) in enumerate(dataloader):
            print(f"\t{i} Images shape: {images.shape}, Masks shape: {masks.shape}, Values in the first mask: {torch.unique(masks[0])}")
            if i >= iterations:
                break

    inspect_batch(train_loader, "Train")
    inspect_batch(val_loader, "Val")
    inspect_batch(test_loader, "Test")

