"""
test_regress.py
---------------
Inference and evaluation script for testing trained STM regression models.

This script performs the following:
1. Loads a trained model (FC, Conv, or Hybrid ResNet).
2. Sets up a data loader for the test dataset.
3. Performs a forward pass to predict values (e.g., interaction strength 'U').
4. If labels are provided, it calculates performance metrics: MAE, MSE, R2, and RMSE.
5. Prints individual predictions alongside ground truth values.

Usage:
    python regression/test_regress.py --data_dir [DATA_DIR] --model_type [MODEL_TYPE] --model_path [MODEL_PATH]
"""
import argparse
import configparser
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import List, Optional
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from common.helper import get_pretrained_model, load_model, seed_everything
from regression.model_regress import SimpleFCRegressorModel, SimpleConvRegressorModel, HybridResNetRegressor
from regression.run_regress import STMTBGDatasetReg, evaluate_regress


def get_test_loader(data_dir: str, batch_size: int=32, input_dim: int=256,
                    label: bool=False,
                    mean: Optional[List[float]]=None,
                    std: Optional[List[float]]=None) -> DataLoader:
    """
    Creates a data loader for testing a regression model.

    Parameters
    -----------
        data_dir: str
            Directory containing the test images.
        batch_size: int
            The batch size for the data loader.
        input_dim: int
            The dimension to resize images to.
        label: bool
            Whether the dataset includes labels.
    
    Returns
    --------

    """
    transform = transforms.Compose([
        transforms.Resize((input_dim, input_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = STMTBGDatasetReg(data_dir, transform=transform, label=label)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the regression model performance on a dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the test dataset.')
    parser.add_argument('--model_type', type=str, choices=['fc', 'conv', 'pretrained'], default='conv', help='Type of model used: "fc" for fully connected, "conv" for convolutional, "pretrained" for pretrained model.') 
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--label', action="store_true", help='Whether the dataset includes labels for evaluation.')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist.")
    
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file {args.model_path} does not exist.")
    
    # Read parameters from config file
    config = configparser.ConfigParser()
    config_path = os.path.join(project_root, 'config', 'config.ini')
    config.read(config_path)

    # General parameters
    input_dim = config.getint('Common Parameters', 'input_size', fallback=256)
    input_channels = config.getint('Common Parameters', 'input_channels', fallback=3)
    device_type = config.get('Common Parameters', 'device', fallback='cpu')
    seed = config.getint('Common Parameters', 'random_seed', fallback=42)
    batch_size = config.getint('Common Parameters', 'batch_size', fallback=64)
    dropout_rate = config.getfloat('Common Parameters', 'dropout_rate', fallback=0.5)
    model_name = config.get('Common Parameters', 'model', fallback='resnet18')
    mean = config.get('Common Parameters', 'mean', fallback='[0.485, 0.456, 0.406]')
    std = config.get('Common Parameters', 'std', fallback='[0.229, 0.224, 0.225]')
    loss_name = config.get('Regression', 'loss_function', fallback='mean_squared_error')

    # Convert mean and std from string to list
    mean = [float(x) for x in mean.strip('[]').split(',')]
    std = [float(x) for x in std.strip('[]').split(',')]

    seed_everything(seed)
    device = torch.device(device_type)

    test_loader = get_test_loader(
        args.data_dir,
        batch_size=batch_size,
        input_dim=input_dim,
        label=args.label,
        mean=mean, std=std
    )

    # Load a pre-trained model
    if args.model_type == 'fc':
        model = SimpleFCRegressorModel(input_dim=input_dim)
    elif args.model_type == 'conv':
        model = SimpleConvRegressorModel(input_channels=input_channels,
                                         dropout_rate=dropout_rate)
    else:  # pretrained
        model = get_pretrained_model(model_name, 1, num_channel=input_channels, device=device, train=False)
        model = HybridResNetRegressor(base_model=model)

    load_model(model, args.model_path)
    model.to(device)
    model.eval()

    if loss_name == 'mean_squared_error':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.L1Loss()   

    if args.label:
        loss, preds, tgts = evaluate_regress(model, test_loader, criterion, device)
        mae = mean_absolute_error(tgts, preds)
        mse = mean_squared_error(tgts, preds)
        r2 = r2_score(tgts, preds)
        rmse = root_mean_squared_error(tgts, preds)
        
        print(f'Test Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}')
        for pred, tgt in zip(preds, tgts):
            print(f'Prediction: {pred.tolist()[0]:.4f}, Target: {tgt.tolist()[0]:.4f}')
    else:
        preds = []
        model.eval()
        with torch.no_grad():
            for img, _ in tqdm(test_loader, desc="Evaluating the dataset", leave=False):
                img = img.to(device)
                outputs = model(img)
                preds.extend(outputs.detach().cpu().numpy().tolist())

        print(f"Preds: {preds}")
