# custom_transformations.py
import random
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from scipy.ndimage import label

class EmphasisCrop(DualTransform):
    """
    Randomly crops the image and mask around a randomly selected object in the mask.
    The crop size is determined by the object's size (largest dimension of its bounding box)
    multiplied by a scale factor (with some random variation). Additional jitter is applied
    to the crop center.

    Args:
        p (float): Probability of applying the transform.
        scale_factor (float): Base scale factor for the object's largest dimension.
        scale_factor_range (tuple): Multiplicative range for scale_factor.
        jitter_ratio (float): Base jitter ratio (fraction of the crop size).
        jitter_ratio_range (tuple): Multiplicative range for jitter_ratio.
        always_apply (bool): If True, apply this transform even if p < 1.
    """

    def __init__(self,
                 p=0.5,
                 scale_factor=10.0,
                 scale_factor_range=(0.9, 1.1),
                 jitter_ratio=0.2,
                 jitter_ratio_range=(0.8, 1.2),
                 always_apply=False):
        super().__init__(always_apply=always_apply, p=p)
        self.scale_factor = scale_factor
        self.scale_factor_range = scale_factor_range
        self.jitter_ratio = jitter_ratio
        self.jitter_ratio_range = jitter_ratio_range

    @property
    def targets_as_params(self):
        # We need the mask to determine the parameters for the crop
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params['mask']

        # Label connected components in the mask
        labeled_mask, num_components = label(mask > 0)
        if num_components == 0:
            # No objects found
            return {"do_crop": False}

        # Pick a random object
        obj_id = random.randint(1, num_components)
        ys, xs = np.where(labeled_mask == obj_id)

        if len(ys) == 0:
            # No pixels found for chosen object (unlikely, but just in case)
            return {"do_crop": False}

        # Compute bounding box of the object
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        obj_height = y_max - y_min + 1
        obj_width = x_max - x_min + 1
        max_dim = max(obj_height, obj_width)

        # Determine random scale factor
        scale_factor_random = self.scale_factor * random.uniform(*self.scale_factor_range)
        crop_size = int(max_dim * scale_factor_random)

        # Use object center as the base center for the crop
        obj_center_y = (y_min + y_max) // 2
        obj_center_x = (x_min + x_max) // 2

        # Determine jitter
        jitter_ratio_random = self.jitter_ratio * random.uniform(*self.jitter_ratio_range)
        jitter_max = int(crop_size * jitter_ratio_random)
        jitter_y = random.randint(-jitter_max, jitter_max)
        jitter_x = random.randint(-jitter_max, jitter_max)

        center_y = obj_center_y + jitter_y
        center_x = obj_center_x + jitter_x

        H, W = mask.shape
        half_size = crop_size // 2

        y1 = max(center_y - half_size, 0)
        x1 = max(center_x - half_size, 0)
        y2 = y1 + crop_size
        x2 = x1 + crop_size

        # Adjust if out of bounds
        if y2 > H:
            diff = y2 - H
            y1 = max(y1 - diff, 0)
            y2 = H
        if x2 > W:
            diff = x2 - W
            x1 = max(x1 - diff, 0)
            x2 = W

        return {
            "do_crop": True,
            "x_min": x1,
            "y_min": y1,
            "x_max": x2,
            "y_max": y2
        }

    def apply(self, img, x_min=0, y_min=0, x_max=None, y_max=None, do_crop=False, **params):
        if do_crop:
            return img[y_min:y_max, x_min:x_max, :]
        return img

    def apply_to_mask(self, mask, x_min=0, y_min=0, x_max=None, y_max=None, do_crop=False, **params):
        if do_crop:
            return mask[y_min:y_max, x_min:x_max]
        return mask

