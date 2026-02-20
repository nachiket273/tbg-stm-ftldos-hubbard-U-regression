"""
grad_cam_regress.py
-------------------
Visualization script using Grad-CAM to interpret regression model predictions.

This script identifies which regions of an STM (Scanning Tunneling Microscopy) 
image most heavily influence the model's predicted regression value (e.g., the 'U' parameter).
It generates a heatmap overlaid on the original image, where "hotter" regions 
indicate high importance for the prediction.

Scientific context:
In FT-LDOS images, Grad-CAM can highlight specific moiré peaks or interference 
patterns that the neural network correlates with specific physical parameters.

Usage:
    python regression/grad_cam_regress.py --data_path [DATA_PATH] --save_path [SAVE_PATH]
    --model_type [MODEL_TYPE] --model_path [MODEL_PATH]
"""
import argparse
import configparser
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
import re
import sys
import torch
from torchvision import transforms
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from common.helper import seed_everything, load_model, get_pretrained_model
from regression.model_regress import HybridResNetRegressor, SimpleConvRegressorModel

matplotlib.use('Agg')


def run_grad_cam_regress(img_path: str, model: torch.nn.Module,
                         target_layer: torch.nn.Module,
                         preprocess: transforms.Compose,
                         save_dir: str,
                         device: str='cpu') -> None:
    """
    Executes Grad-CAM on a single image and saves the resulting visualization.

    Parameters
    -----------
        img_path: str
            Path to the input image file.
        model: torch.nn.Module
            The trained regression model.
        target_layer: torch.nn.Module
            The specific layer (usually last conv layer) to compute gradients from.
        preprocess: transforms.Compose
            Torchvision transform pipeline.
        save_dir: str
            Directory to store the output heatmap images.
        device: str
            'cpu' or 'cuda'.
    """
    # 1. Keep the original image for the final overlay
    orig_img_pil = Image.open(img_path).convert('RGB')
    orig_width, orig_height = orig_img_pil.size

    # 2. Preprocess for the model
    img_l = Image.open(img_path).convert('L')
    input_tensor = preprocess(img_l).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.item()

    target_class = None
    base_name = os.path.basename(img_path)

    # match = re.search(r'U(\d+)', base_name)
    match = re.search(r'U(\d+.\d+)', base_name)
    if match:
        target_class = float(match.group(1))

    # Grad-CAM
    with torch.enable_grad():
        cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[RawScoresOutputTarget()])[0, :]
    grayscale_cam_highres = cv2.resize(grayscale_cam, (orig_width, orig_height))

    # 5. Convert original PIL image to float array (0-1) for show_cam_on_image
    orig_img_array = np.array(orig_img_pil).astype(np.float32) / 255.0

    # 6. Overlay on the original resolution image
    visualization = show_cam_on_image(orig_img_array, grayscale_cam_highres,
                                      use_rgb=True, image_weight=0.6)

    # generate save path
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'grad_cam_{base_name}')

    # Overlay the heatmap on the original image
    name = os.path.splitext(base_name)[0]
    save_name = f"{name}_gradcam_U_{target_class if target_class is not None else 'unknown'}_pred{pred:.3f}.png"
    save_path = os.path.join(save_dir, save_name)
    plt.figure(figsize=(12, 12), dpi=300)
    plt.imsave(save_path, visualization, dpi=300)
    plt.close()


# Example Usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grad CAM Visualization Regression.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, required=True, help='Path to the folder with input images.')
    parser.add_argument('--save_path', type=str, default='guided_backprop', help='Path to save the output visualization.')
    parser.add_argument('--model_type', type=str, choices=['fc', 'conv', 'pretrained'], default='pretrained', help='Type of model to use.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to a pre-trained model.')
    args = parser.parse_args()

    # Read parameters from config file
    config = configparser.ConfigParser()
    config_path = os.path.join(project_root, 'config', 'config.ini')
    config.read(config_path)

    input_dim = config.getint('Common Parameters', 'input_size', fallback=256)
    input_channels = config.getint('Common Parameters', 'input_channels', fallback=3)
    device_type = config.get('Common Parameters', 'device', fallback='cpu')
    seed = config.getint('Common Parameters', 'random_seed', fallback=42)
    dropout_rate = config.getfloat('Common Parameters', 'dropout_rate', fallback=0.5)
    model_name = config.get('Common Parameters', 'model', fallback='resnet18')
    mean = config.get('Common Parameters', 'mean', fallback='[0.485, 0.456, 0.406]')
    std = config.get('Common Parameters', 'std', fallback='[0.229, 0.224, 0.225]')

    # Convert mean and std from string to list
    mean = [float(x) for x in mean.strip('[]').split(',')]
    std = [float(x) for x in std.strip('[]').split(',')]

    # Seed all randomness
    seed_everything(seed)

    device = torch.device(device_type)
    print(f'Using device: {device}')

    if os.path.exists(args.save_path) is False:
        os.makedirs(args.save_path)
    
    if args.model_type == 'pretrained':
    # Load a pre-trained model
        model = get_pretrained_model(model_name, num_classes=1, num_channel=input_channels,
                                     device=device, train=False)
        model = HybridResNetRegressor(base_model=model)
        load_model(model, args.model_path, device)
        model.to(device)
        model.eval()
    elif args.model_type == 'conv':
        model = SimpleConvRegressorModel(input_channels=input_channels,
                                         dropout_rate=dropout_rate)
        load_model(model, args.model_path, device)
        model.to(device)
        model.eval()
    else:
        raise NotImplementedError("Only 'pretrained' and 'conv' model type is implemented in this example.")
    
    preprocess = transforms.Compose([
        transforms.Resize((input_dim, input_dim)),
        # HighPassFilter(4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Define the target layer for Grad-CAM (the last convolutional layer)
    if args.model_type == 'pretrained':
        target_layer = model.cnn[-2][-1].conv2
    elif args.model_type == 'conv':
        target_layer = model.features[-4]  # Last Conv2d layer in SimpleConvRegressorModel

    # Run Grad-CAM for each image in the specified folder
    for img_file in tqdm(os.listdir(args.data_path), leave=False, desc="Processing input images"):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(args.data_path, img_file)
            run_grad_cam_regress(img_path, model, target_layer, preprocess,
                                 input_dim=input_dim, save_dir=args.save_path,
                                 device=device)
