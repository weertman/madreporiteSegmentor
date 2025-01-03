# inference.py

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch

# Import from the backend
from inference_backend import InferenceSession

def run_inference_programmatically(
    model_path: str,
    input_images: list,         # A list of image paths
    output_dir: str = ".",
    threshold: float = 0.5,
    device: str = None
):
    """
    A function that can be called from Python or Jupyter to run inference on many images
    without re-loading the model for each.

    Args:
      model_path (str): Path to the .pth checkpoint (inside 'checkpoints/').
      input_images (list): List of paths to images to segment.
      output_dir (str): Directory to save each segmentation mask.
      threshold (float): Probability threshold for binarization.
      device (str): e.g. 'cpu' or 'cuda:0'. If None, use config or fallback.

    Returns:
      A dictionary mapping { image_path -> mask_array (np.uint8) }
    """
    # 1) Build an InferenceSession (loads the model once)
    session = InferenceSession(
        model_path=Path(model_path),
        device_str=device,
        threshold=threshold
    )

    # 2) Make sure output_dir is a directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) Predict each image
    results_dict = {}
    for img_path in input_images:
        img_path = Path(img_path)
        mask = session.predict_image(img_path)
        results_dict[str(img_path)] = mask

        # 4) Save the mask
        # e.g. "my_image.jpg" -> "my_image_mask.png"
        out_name = f"{img_path.stem}_mask.png"
        out_path = output_dir / out_name

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil.save(out_path)
        print(f"Saved mask for {img_path} to {out_path}")

    return results_dict


def run_inference_cli():
    """
    Command-line interface: parse arguments, run inference on multiple images, save masks.
    """
    parser = argparse.ArgumentParser(description="CLI for segmentation inference on multiple images.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model.pth (inside 'checkpoints').")
    parser.add_argument("--input_images", type=str, nargs='+', required=True,
                        help="One or more image paths to segment.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory where masks will be saved.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binarizing masks.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference, e.g. 'cpu' or 'cuda:0'.")
    args = parser.parse_args()

    # Just call the programmatic function
    run_inference_programmatically(
        model_path=args.model_path,
        input_images=args.input_images,
        output_dir=args.output_dir,
        threshold=args.threshold,
        device=args.device
    )

if __name__ == "__main__":
    run_inference_cli()
