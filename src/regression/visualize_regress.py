"""
visualize_regress.py
--------------------
Interpretability script using Guided Backpropagation for regression models.

This script performs the following:
1. Loads a trained regression model (e.g., Conv or ResNet-based).
2. Uses Guided Backpropagation to compute gradients of the output with respect 
   to the input image pixels.
3. Generates high-resolution saliency maps showing which physical features 
   (e.g., moiré lattice points) the model is "looking at" to determine the 'U' value.
4. Saves high-quality grayscale saliency maps and color-mapped overlays.

Usage:
    python regression/visualize_regress.py --data_path [DATA_PATH] --save_path [SAVE_PATH]
    --model_type [MODEL_TYPE] --model_path [MODEL_PATH]
"""
import argparse
import configparser
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
from pytorch_grad_cam import GuidedBackpropReLUModel
import sys
import torch
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from common.helper import load_model, get_pretrained_model, seed_everything
from regression.model_regress import SimpleFCRegressorModel, SimpleConvRegressorModel, HybridResNetRegressor

matplotlib.use('Agg')


def save_large_image(arr: np.ndarray, out_path: str,
                     cmap: str="magma", figsize: Tuple[int, int]=(10, 10),
                     dpi: int=300) -> None:
    """
    Saves a NumPy array as a high-resolution image using a specific colormap.

    Parameters
    -----------
        arr: np.ndarray
            2D array of pixel intensities.
        out_path: str
            File path to save the image.
        cmap: str
            Matplotlib colormap name.
        figsize: tuple(int, int)
            Figure dimensions in inches.
        dpi: int
            Dots per inch for resolution control.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(arr, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()


def overlay_heatmap(rgb: np.ndarray, saliency: np.ndarray, out_path: str,
                    alpha: float=0.5, figsize: Tuple[int, int]=(10, 10),
                    dpi: int=300) -> None:
    """
    Overlays a saliency heatmap on top of the original image.

    Parameters
    -----------
        rgb: np.ndarray
            Original image (normalized 0-1).
        saliency: np.ndarray
            The saliency map (normalized 0-1).
        out_path: str
            Destination file path.
        alpha: float
            Transparency level for the overlay.
        figsize: tuple(int, int)
            Figure dimensions in inches.
        dpi: int
            Dots per inch for resolution control
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(rgb)
    plt.imshow(saliency, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()


def guided_backprop_visualization(model: torch.nn.Module, image_path: str,
                                  gb_model: GuidedBackpropReLUModel,
                                  transform: transforms.Compose,
                                  save_path: str, input_dim: int,
                                  device: str='cpu') -> None:
    """
    Generates, processes, and saves Guided Backpropagation visualizations for a regression prediction.
    
    This function computes the gradients of the model's output with respect to the input pixels,
    normalizes them to highlight the most "important" features, and saves three versions:
    1. Grayscale saliency map.
    2. RGB gradient map.
    3. Heatmap overlaid on the original image.

    Args:
        model: torch.nn.Module
            The trained regression model.
        image_path: str
            Path to the source STM image.
        gb_model: GuidedBackpropReLUModel
            Initialized GuidedBackpropReLUModel wrapper.
        transform: transforms.Compose
            The preprocessing pipeline (Resize, ToTensor, Normalize).
        save_path: str
            Directory where output images will be stored.
        input_dim: int
            The spatial resolution (H/W) for resizing.
        device: str
            The computing device ('cpu' or 'cuda').
    """
    img = Image.open(image_path).convert('L')
    rgb_img = Image.open(image_path).convert('RGB')
    rgb_img = np.array(rgb_img.resize((input_dim, input_dim))).astype(np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(device)

    # predict
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).item()

    # Guided Backpropagation
    model.zero_grad()
    input_tensor.requires_grad = True
    with torch.enable_grad():
        grads = gb_model(input_tensor, target_category=None)

    # Ensure correct shape: [C, H, W] → [H, W, C]
    if isinstance(grads, torch.Tensor):
        grads = grads.detach().cpu().numpy()
    if grads.ndim == 4:
        grads = grads[0]
    if grads.shape[0] in [1, 3]:
        grads = np.transpose(grads, (1, 2, 0))

    # Normalize gradients for visualization
    grads = np.abs(grads)
    grads /= grads.max() + 1e-8

    # Create grayscale version
    grads_gray = grads.mean(axis=-1)
    grads_gray = (grads_gray - grads_gray.min()) / (grads_gray.max() - grads_gray.min() + 1e-12)

    name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(save_path, exist_ok=True)

    save_large_image(grads_gray, os.path.join(save_path, f"{name}_guided_gray_pred{pred:.3f}.png"))
    save_large_image(grads, os.path.join(save_path, f"{name}_guided_rgb_pred{pred:.3f}.png"), cmap=None)
    overlay_heatmap(rgb_img, grads_gray, os.path.join(save_path, f"{name}_guided_overlay_pred{pred:.3f}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Guided Backpropagation Visualization',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, required=True, help='Path to the folder with input images.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output visualization.')
    parser.add_argument('--model_type', type=str, choices=['fc', 'conv', 'pretrained'],
                        default='fc', help='Type of model to use.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to a pre-trained model.')
    args = parser.parse_args()

    # Read parameters from config file
    config = configparser.ConfigParser()
    config_path = os.path.join(project_root, 'config', 'config.ini')
    config.read(config_path)

    input_dim = config.getint('Common Parameters', 'input_size', fallback=256)
    input_channels = config.getint('Common Parameters', 'input_channels', fallback=3)
    device_type = config.get('Common Parameters', 'device', fallback='cpu')
    model_name = config.get('Common Parameters', 'model', fallback='resnet18')
    mean = config.get('Common Parameters', 'mean', fallback='[0.485, 0.456, 0.406]')
    std = config.get('Common Parameters', 'std', fallback='[0.229, 0.224, 0.225]')
    dropout_rate = config.getfloat('Common Parameters', 'dropout_rate', fallback=0.5)
    seed = config.getint('Common Parameters', 'random_seed', fallback=42)

    seed_everything(seed)

    # Convert mean and std from string to list
    mean = [float(x) for x in mean.strip('[]').split(',')]
    std = [float(x) for x in std.strip('[]').split(',')]

    device = torch.device(device_type)
    print(f'Using device: {device}')

    # Load a pre-trained model
    if args.model_type == 'fc':
        model = SimpleFCRegressorModel(input_dim=input_dim)
    elif args.model_type == 'conv':
        model = SimpleConvRegressorModel(input_channels=input_channels,
                                         dropout_rate=dropout_rate)
    else:  # pretrained
        model = get_pretrained_model(model_name, 1, num_channel=input_channels, device=device)
        model = HybridResNetRegressor(base_model=model)

    load_model(model, args.model_path)
    model.to(device)
    model.eval()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((input_dim, input_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    gb_model = GuidedBackpropReLUModel(model=model, device=device)

    # Get all image paths from the specified directory
    image_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path)
                   if os.path.isfile(os.path.join(args.data_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in tqdm(image_files, leave=False, desc="Processing images"):
        guided_backprop_visualization(model, image_path, gb_model, transform,
                                      args.save_path, input_dim=input_dim, device=device)
