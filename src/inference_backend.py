# inference_backend.py

import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import cv2  # We'll use OpenCV for video reading

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Reuse existing project modules:
from unet import PVTv2UNetSegmentationModel
from resnet import PVTv2ResNetSegmentationModel


def load_config_for_model_path(model_path: Path):
    """
    Given the path to best_model.pth (in 'checkpoints'),
    go one directory up and look for 'config.yaml'.
    If found, load it. Otherwise return {}.
    """
    run_dir = model_path.parent.parent  # one level above 'checkpoints/'
    config_file = run_dir / "config.yaml"

    if not config_file.exists():
        print(f"No config.yaml found at {config_file}. Returning empty config.")
        return {}

    print(f"Found config.yaml at {config_file}. Loading...")
    with open(config_file, 'r') as cf:
        config_data = yaml.safe_load(cf)
    return config_data


def build_inference_transform(input_shape=(640, 640)):
    """
    Builds a minimal transform pipeline for inference.
    Adjust as needed to match your eval-time transforms (normalize, etc.).
    """
    height, width = input_shape
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def create_model_from_config(config_dict: dict, model_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Builds and loads a model from config_dict (parsed from config.yaml).
    model_path is the .pth file. If config is partial or missing fields,
    we use some defaults.
    """

    # Defaults if missing
    decoder_type = config_dict.get("model", {}).get("decoder_type", "unet")
    backbone = config_dict.get("model", {}).get("backbone", "pvt_v2_b0")
    pretrained = config_dict.get("model", {}).get("pretrained", False)
    freeze_backbone = config_dict.get("model", {}).get("freeze_backbone", False)
    decoder_init_method = config_dict.get("model", {}).get("decoder_init_method", "default")
    decoder_blocks = config_dict.get("model", {}).get("decoder_blocks", 5)
    input_shape = config_dict.get("model", {}).get("input_shape", (640, 640))
    num_classes = config_dict.get("model", {}).get("num_classes", 1)

    # Build model
    if decoder_type == "unet":
        model = PVTv2UNetSegmentationModel(
            model_name=f"OpenGVLab/{backbone}",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            decoder_init_method=decoder_init_method,
            input_shape=input_shape
        )
    elif decoder_type == "resnet":
        model = PVTv2ResNetSegmentationModel(
            model_name=f"OpenGVLab/{backbone}",
            num_classes=num_classes,
            pretrained=pretrained,
            decoder_blocks=decoder_blocks,
            freeze_backbone=freeze_backbone,
            decoder_init_method=decoder_init_method
        )
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    model.to(device)
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    print("Model loaded from checkpoint. Ready for inference.")
    return model


class InferenceSession:
    """
    An object that loads the model once, then can be used to run inference on
    many images or frames without re-loading the model each time.
    Supports:
      - Single image path or np.array
      - Multiple images (paths or np.arrays)
      - Video file (reads frames)
      - Batch processing
    """

    def __init__(self, model_path: Path, device_str: str = None, threshold: float = 0.5, batch_size: int = 4):
        """
        model_path: path to the .pth checkpoint (inside 'checkpoints/')
        device_str: e.g. 'cuda:0' or 'cpu'
        threshold: probability threshold for binarizing predictions
        batch_size: how many images/frames to process at once
        """
        self.model_path = model_path
        self.threshold = threshold
        self.batch_size = batch_size

        # 1) Load config
        self.config_dict = load_config_for_model_path(model_path)

        # 2) Determine device
        if device_str is not None:
            self.device_str = device_str
        else:
            # If user didn't supply device, check config or fallback to 'cuda:0'
            self.device_str = self.config_dict.get("training", {}).get("device", "cuda:0")

        self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
        print(f"InferenceSession using device: {self.device}")

        # 3) Build and store the model
        self.model = create_model_from_config(
            self.config_dict,
            model_path,
            self.device
        )

        # 4) Build transforms
        model_input_shape = self.config_dict.get("model", {}).get("input_shape", (640, 640))
        self.transform = build_inference_transform(input_shape=model_input_shape)

    def predict_image(self, image_input) -> np.ndarray:
        """
        Run inference on a SINGLE image input, returning a 0/1 np.array mask.

        image_input can be:
          - A path (str or Path) to an image file
          - A np.array (H, W, 3) image
        """
        img_np = self._load_image_array(image_input)
        # Transform and predict
        tensor = self._transform_one(img_np)
        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0, 0]

        mask_binary = (probs >= self.threshold).cpu().numpy().astype(np.uint8)
        return mask_binary

    def predict_images(self, inputs, is_video=False) -> list:
        """
        Run inference on MULTIPLE images (paths or np.arrays) or a single video path.

        If is_video==True, 'inputs' should be a path to a video file (str or Path).
        Otherwise, 'inputs' can be a list of:
          - image file paths
          - np.array images
        Returns a list of predicted masks (each is a np.uint8 array).

        Uses batch processing for efficiency: we accumulate up to self.batch_size images
        before passing them through the model.
        """
        if is_video:
            # 'inputs' is a single path to a video
            frame_arrays = self._read_video_frames(inputs)
            return self._batch_inference(frame_arrays)
        else:
            # 'inputs' is a list of images (paths or arrays)
            image_arrays = []
            for inp in inputs:
                img_np = self._load_image_array(inp)
                image_arrays.append(img_np)
            return self._batch_inference(image_arrays)

    # -------------------------------------------------------------------------
    # NEW METHODS FOR TWO-PASS REFINEMENT
    # -------------------------------------------------------------------------

    @staticmethod
    def _find_center_of_largest_cc(mask: np.ndarray) -> tuple:
        """
        Find the center of the largest connected component in a 2D {0,1} mask.
        Returns (row, col). If no foreground, returns the image center.
        """
        mask_binary = (mask > 0).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        if num_labels <= 1:
            # No foreground => fallback to image center
            return (mask.shape[0] // 2, mask.shape[1] // 2)

        largest_area = 0
        largest_label = 1
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_label = label

        cx, cy = centroids[largest_label]  # (x, y)
        return (int(cy), int(cx))  # (row, col)

    @staticmethod
    def _keep_largest_cc(mask: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected component in a 2D {0,1} mask,
        returning a {0,1} mask of the same shape.
        """
        mask_binary = (mask > 0).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(mask, dtype=mask.dtype)

        largest_area = 0
        largest_label = 1
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_label = label

        out_mask = np.zeros_like(mask_binary, dtype=mask_binary.dtype)
        out_mask[labels_im == largest_label] = 1
        return out_mask.astype(mask.dtype)

    def predict_image_with_refinement(self, image_input, refinement_ratio=0.2268, threshold=None) -> np.ndarray:
        """
        Two-pass inference:
          1) Predict entire image => binarize with threshold
          2) Find largest CC center, crop around it by refinement_ratio
          3) Second pass on subregion => threshold
          4) Keep largest CC in subregion
          5) Resize subregion's refined mask back and paste into a mask the size of the original

        Returns a 2D np.array {0,1} matching the original image shape.
        """
        if threshold is None:
            threshold = self.threshold

        # Load full image (RGB or BGR; we assume it's consistent with .predict_image)
        img_np = self._load_image_array(image_input)
        h, w = img_np.shape[:2]

        # First-pass inference (mask is 0/1 at model's input_size, then scaled if needed).
        first_pass_mask = self.predict_image(img_np)  # shape might be ~ the model's input, but returned as 0/1
        # If needed, you could do direct transform logic here, but using .predict_image is simpler.

        # Find center of largest CC in the full-res mask
        # Because .predict_image might have returned exactly the model's input-size mask,
        # we assume the transform un-distorted it to (h, w).
        # If the model's input_shape differs from your original, you can do an explicit resize.
        # But in this pipeline, we typically keep model_input_shape == original shape or do a known ratio.

        # For safety, let's ensure the mask is the same size as original (h, w) if needed:
        if first_pass_mask.shape[0] != h or first_pass_mask.shape[1] != w:
            first_pass_mask = cv2.resize(first_pass_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        center = self._find_center_of_largest_cc(first_pass_mask)

        # Crop bounds
        half_crop_w = int(w * refinement_ratio / 2)
        half_crop_h = int(h * refinement_ratio / 2)

        x0 = center[1] - half_crop_w
        x1 = center[1] + half_crop_w
        y0 = center[0] - half_crop_h
        y1 = center[0] + half_crop_h

        # Clamp to edges
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)

        # Second pass: predict on cropped subregion
        sub_image = img_np[y0:y1, x0:x1]
        sub_mask = self.predict_image(sub_image)
        # If needed, ensure sub_mask is also sub_image's shape => typically yes.

        # Keep largest CC in subregion
        sub_mask = self._keep_largest_cc(sub_mask)

        # Resize sub_mask to exactly the subregion size in case the model transform changed it
        cropW = (x1 - x0)
        cropH = (y1 - y0)
        if sub_mask.shape[0] != cropH or sub_mask.shape[1] != cropW:
            sub_mask = cv2.resize(sub_mask, (cropW, cropH), interpolation=cv2.INTER_NEAREST)

        # Paste sub_mask into a black mask of the full size
        refined_mask = np.zeros((h, w), dtype=np.uint8)
        refined_mask[y0:y1, x0:x1] = sub_mask

        return refined_mask

    # -------------------------------------------------
    # INTERNAL UTILS
    # -------------------------------------------------

    def _load_image_array(self, image_input):
        """
        Load a single image as a np.array, from either a path or direct array.
        """
        if isinstance(image_input, (str, Path)):
            # read from file
            img_path = Path(image_input)
            img_pil = Image.open(img_path).convert("RGB")
            return np.array(img_pil)
        elif isinstance(image_input, np.ndarray):
            # assume it's already H,W,3 or H,W
            return image_input
        else:
            raise TypeError("image_input must be path-like or a np.ndarray")

    def _transform_one(self, img_np):
        """
        Apply Albumentations transform to a single image array,
        return a (1, C, H, W) tensor.
        """
        transformed = self.transform(image=img_np)
        img_tensor = transformed["image"].unsqueeze(0).to(self.device)  # (1,C,H,W)
        return img_tensor

    def _batch_inference(self, image_arrays) -> list:
        """
        Given a list of image arrays, do batch processing.
        Returns list of binary masks in the same order.
        """
        masks = []
        batch_buffer = []
        # We'll store (index, array) so we can reconstruct output in the correct order
        for idx, img_np in enumerate(image_arrays):
            batch_buffer.append((idx, img_np))
            # If we reached batch_size, or we're at the end, run inference
            if len(batch_buffer) == self.batch_size:
                masks_batch = self._run_batch(batch_buffer)
                masks.extend(masks_batch)
                batch_buffer = []

        # handle leftover
        if batch_buffer:
            masks_batch = self._run_batch(batch_buffer)
            masks.extend(masks_batch)

        # sort them by idx, then return only the mask
        masks_sorted = sorted(masks, key=lambda x: x[0])
        return [m[1] for m in masks_sorted]  # strip index, keep mask

    def _run_batch(self, batch_buffer):
        """
        Run inference on a batch of images stored as (index, np.array).
        Returns list of (index, mask).
        """
        # 1) transform all
        tensors = []
        idxes = []
        for (idx, img_np) in batch_buffer:
            transformed = self.transform(image=img_np)
            tensors.append(transformed["image"])
            idxes.append(idx)

        # 2) stack into (B,C,H,W)
        batch_tensor = torch.stack(tensors, dim=0).to(self.device)

        # 3) forward pass
        with torch.no_grad():
            logits = self.model(batch_tensor)  # (B,1,H,W)
            probs = torch.sigmoid(logits).squeeze(1)  # (B,H,W)

        # 4) threshold & convert
        batch_masks = (probs >= self.threshold).cpu().numpy().astype(np.uint8)

        # 5) store as (idx, mask)
        results = [(idxes[i], batch_masks[i]) for i in range(len(idxes))]
        return results

    def _read_video_frames(self, video_path):
        """
        Read frames from a video file using OpenCV,
        returning a list of np arrays (BGR).
        We can convert to RGB if needed. For large videos,
        you might want to process them in streaming fashion
        instead of loading them all into memory!
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video {video_path}")

        frame_arrays = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame is BGR, convert to RGB if needed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_arrays.append(frame_rgb)

        cap.release()
        return frame_arrays
