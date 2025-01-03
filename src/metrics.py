# metrics.py
import torch
import torch.nn.functional as F


class DiceCoefficient:
    """
    Computes the Dice Coefficient for binary segmentation.

    Dice = 2 * TP / (2 * TP + FP + FN)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: (B, 1, H, W) or (B, H, W)
        targets: (B, H, W) binary {0,1}

        - We'll apply sigmoid + threshold to preds if they are raw logits.
        """
        if preds.dim() == 4:
            # (B, 1, H, W)
            preds = preds.squeeze(1)  # make it (B, H, W) for easier handling
        # Apply sigmoid to get probabilities
        preds = torch.sigmoid(preds)
        # Threshold at 0.5
        preds = (preds > 0.5).long()

        # Flatten to compute counts easily
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1).long()

        tp = torch.sum(preds_flat * targets_flat)
        fp = torch.sum(preds_flat * (1 - targets_flat))
        fn = torch.sum((1 - preds_flat) * targets_flat)

        self.tp += tp.item()
        self.fp += fp.item()
        self.fn += fn.item()

    def compute(self):
        numerator = 2 * self.tp
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return 1.0  # If no foreground at all in both pred & target, consider Dice=1
        return numerator / denominator


class IoU:
    """
    Computes the Intersection over Union (IoU) for binary segmentation.

    IoU = TP / (TP + FP + FN)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: (B, 1, H, W) or (B, H, W)
        targets: (B, H, W) binary {0,1}

        - We'll apply sigmoid + threshold to preds if they are raw logits.
        """
        if preds.dim() == 4:
            preds = preds.squeeze(1)  # (B, H, W)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).long()

        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1).long()

        tp = torch.sum(preds_flat * targets_flat)
        fp = torch.sum(preds_flat * (1 - targets_flat))
        fn = torch.sum((1 - preds_flat) * targets_flat)

        self.tp += tp.item()
        self.fp += fp.item()
        self.fn += fn.item()

    def compute(self):
        denominator = (self.tp + self.fp + self.fn)
        if denominator == 0:
            return 1.0  # If no positives in both pred and target, consider IoU=1
        return self.tp / denominator


import torch

class ProbMaxInMask:
    """
    Computes a variant of the probability that the 'hottest' pixel in the prediction
    is "correct" — i.e., if the ground-truth mask is non-empty, the hottest pixel
    should lie inside the mask; if the ground-truth mask is empty (all zeros),
    then the model's max probability should be below a chosen threshold (pcut).

    We define:
        Pin = TP / (TP + FP)

    Where "TP" means:
      - If mask is non-empty and hottest pixel is in the mask, or
      - If mask is empty and the model's max probability < pcut,
    and "FP" means the opposite.

    Implementation details:
      1) We assume `preds` are raw logits (same as Dice/IoU). We'll apply sigmoid internally.
      2) For each image:
         - If sum of target==0 (no mask), we check if max(prob)<pcut. If yes => TP, else => FP.
         - If sum of target>0 (mask exists), we find the hottest pixel's location and check if that pixel belongs to the mask.
      3) We do one increment in the denominator per image.

    """

    def __init__(self, pcut: float = 0.5):
        """
        Args:
            pcut (float): Probability threshold used to decide
                          if the model is "highlighting" a pixel for empty masks.
        """
        self.pcut = pcut
        self.reset()

    def reset(self):
        # For each image, we do exactly 1 increment in the denominator.
        self.tp_fp = 0
        self.tp = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds:   (B, 1, H, W) or (B, H, W) raw logits.
            targets: (B, H, W) binary mask {0,1}.
        """
        if preds.dim() == 4:
            preds = preds.squeeze(1)  # => (B, H, W)

        probs = torch.sigmoid(preds)               # => (B, H, W)
        B, H, W = probs.shape
        flat_probs = probs.view(B, -1)             # => (B, H*W)
        flat_targets = targets.view(B, -1).long()  # => (B, H*W)

        for i in range(B):
            self.tp_fp += 1  # Each image contributes once to the denominator

            # Check if mask is empty
            if flat_targets[i].sum() == 0:
                # No mask. If max(prob) < pcut => "correct negative" => tp++
                max_prob = flat_probs[i].max()
                if max_prob < self.pcut:
                    self.tp += 1
            else:
                # Mask is not empty. We locate the single hottest pixel:
                max_idx = flat_probs[i].argmax()
                # If that pixel is in the mask => tp++
                if flat_targets[i, max_idx] == 1:
                    self.tp += 1

    def compute(self):
        if self.tp_fp == 0:
            return 0.0
        return self.tp / self.tp_fp
