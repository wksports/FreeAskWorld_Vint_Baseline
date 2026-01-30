#!/usr/bin/env python3
"""ViNT Inference Script (single GPU, vint model only)

Features:
- Load vint model configuration and checkpoint from deployment/config/models.yaml
- Read local images (as context) and one goal image, run inference and print output
- No ROS required, suitable for offline validation of trained vint.pth
"""
# Import necessary libraries
import os
import argparse
import yaml
import torch
from PIL import Image as PILImage
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF

# Import ViNT model and data utilities
from vint_train.models.vint.vint import ViNT
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO


def load_model(model_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    """Load ViNT model from checkpoint (single GPU)

    Args:
        model_path: Checkpoint file path (.pth)
        config: Model configuration dict (from yaml)
        device: Target device (cuda or cpu)

    Returns:
        ViNT model instance with loaded weights
    """
    # Build ViNT model from config
    model = ViNT(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    # Load checkpoint from disk
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Extract model weights state_dict
    state_dict = checkpoint["model"].state_dict()
    # Load weights to model (strict=False allows partial mismatch)
    model.load_state_dict(state_dict, strict=False)
    # Move model to target device
    model.to(device)
    return model


def to_numpy(tensor):
    """Convert torch Tensor to numpy array (move to CPU and detach first).

    Args:
        tensor: torch Tensor

    Returns:
        numpy array
    """
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs, image_size, center_crop=False):
    """Convert PIL image(s) to normalized torch Tensor (ViNT input format).

    Args:
        pil_imgs: Single PIL.Image or list of PIL.Image
        image_size: Target size [width, height]
        center_crop: Whether to apply center crop (default False)

    Returns:
        Normalized torch Tensor with shape (1, 3*(context_size+1), H, W)
    """
    # Define image transforms: ToTensor + Normalize (ImageNet mean and std)
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Ensure input is a list for uniform processing
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        # Optional center crop to match model's expected aspect ratio
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        # Resize to configured image_size
        pil_img = pil_img.resize(image_size)
        # ToTensor + Normalize (consistent with training)
        transf_img = transform_type(pil_img)
        # Add batch dimension
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    # Concatenate on channel dimension (returns shape ready for model input)
    return torch.cat(transf_imgs, dim=1)


def resolve_rel(base_path: str, rel: str) -> str:
    """Resolve relative path to absolute path.

    Args:
        base_path: Base path
        rel: Relative path

    Returns:
        Absolute path
    """
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(base_path, rel))


def load_pil(path: str) -> PILImage.Image:
    """Load image from file and convert to RGB mode.

    Args:
        path: Image file path

    Returns:
        PIL.Image object (RGB mode)
    """
    return PILImage.open(path).convert("RGB")


def main():
    """Main function: parse args, load model, read images, run inference and print results."""
    # ---------- Parse command line arguments ----------
    parser = argparse.ArgumentParser(description="ViNT single GPU inference script")
    parser.add_argument("--context-dir", "-c", default="../topomaps/images", help="Context images directory")
    parser.add_argument("--goal", "-g", default=None, help="Goal image path (default: last image in dir)")
    parser.add_argument("--num-context", "-n", type=int, default=None, help="Context frames count (overrides config)")
    args = parser.parse_args()

    # ---------- Load model configuration ----------
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    models_yaml = os.path.join(base_dir, "config", "models.yaml")
    with open(models_yaml, "r") as f:
        model_paths = yaml.safe_load(f)

    # Get vint model config path and checkpoint path
    model_entry = model_paths["vint"]
    models_yaml_dir = os.path.dirname(models_yaml)
    model_config_path = resolve_rel(models_yaml_dir, model_entry["config_path"])
    ckpt_path = resolve_rel(models_yaml_dir, model_entry["ckpt_path"])

    # Check if checkpoint file exists
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load model configuration parameters
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    # Determine context size (can be overridden by command line args)
    context_size = model_params.get("context_size", 5)
    if args.num_context is not None:
        context_size = args.num_context
    # ViNT expects context_size + 1 images (including current frame)
    num_images = context_size + 1

    # ---------- Select device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- Load model ----------
    print(f"Loading model from {ckpt_path} with config {model_config_path}")
    model = load_model(ckpt_path, model_params, device)
    # Switch to evaluation mode (disable dropout etc.)
    model.eval()

    # ---------- Load context and goal images ----------
    # Resolve absolute path of context images directory
    ctx_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), args.context_dir))
    if not os.path.isdir(ctx_dir):
        raise FileNotFoundError(f"Context directory not found: {ctx_dir}")

    # Get all image files in directory and sort by filename
    imgs = sorted([p for p in os.listdir(ctx_dir) if p.lower().endswith((".jpg", ".png", ".jpeg"))], key=lambda x: x)
    # Check if enough images available
    if len(imgs) < num_images:
        raise RuntimeError(f"Not enough images in {ctx_dir} (need {num_images}, found {len(imgs)})")

    # Select last N images from directory as context
    chosen = imgs[-num_images:]
    context_imgs = [load_pil(os.path.join(ctx_dir, p)) for p in chosen]

    # Load goal image (use last image in directory if not specified)
    if args.goal:
        goal_img = load_pil(args.goal)
    else:
        goal_img = load_pil(os.path.join(ctx_dir, imgs[-1]))

    # ---------- Image preprocessing ----------
    # Get model's expected image size
    image_size = model_params.get("image_size", [85, 64])

    # Convert context and goal images to normalized Tensors
    obs_tensor = transform_images(context_imgs, image_size)
    goal_tensor = transform_images(goal_img, image_size)

    # Move Tensors to target device (CPU/GPU)
    obs_tensor = obs_tensor.to(device)
    goal_tensor = goal_tensor.to(device)

    # ---------- Forward inference ----------
    # Disable gradient computation to save memory
    with torch.no_grad():
        # Execute model forward pass
        out = model(obs_tensor, goal_tensor)

    # ---------- Print output ----------
    # Output is typically a (dist_pred, action_pred) tuple
    if isinstance(out, tuple) or isinstance(out, list):
        for i, o in enumerate(out):
            print(f"Output[{i}] shape: {tuple(o.shape)}")
            print(to_numpy(o))
    else:
        print("Model output:")
        print(to_numpy(o))


if __name__ == "__main__":
    main()
