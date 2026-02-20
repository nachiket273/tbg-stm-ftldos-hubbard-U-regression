"""
run_regress.py
--------------
Core training and data management utilities for STM regression tasks.

This module provides:
1. STMTBGDatasetReg: A custom PyTorch Dataset that extracts physical parameters
   (like the 'U' value) directly from image filenames.
2. get_data_loaders_regress: A robust data pipeline that handles stratified-like 
   splitting, automated normalization calculation, and DataLoader creation.
3. train_regress/evaluate_regress: Standard training and validation loops 
   optimized for regression metrics.
4. plot_stats_regress: Visualization of training progress across multiple metrics.
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import re
import sys
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import List, Tuple, Optional, Dict, Any, Unions

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from common.helper import get_split_indices


class STMTBGDatasetReg(Dataset):
    """
    Dataset class for Scanning Tunneling Microscopy images used in regression.
    Extracts the target 'U' value from filenames using regular expressions.
    """
    def __init__(self, directory: str, transform: Optional[Any]=None, label: bool=True) -> None:
        """
        Args:
            directory: str
                Path to the folder containing image files.
            transform: Optional[Any]
                Torchvision transformations to apply to images.
            label: bool
                If True, attempts to parse the 'U' value from the filename.
        """
        self.directory = directory
        self.transform = transform
        self.image_paths, self.dos_paths, self.band_paths = [], [], []
        for f in os.listdir(directory):
            if f.endswith('.png'):
                self.image_paths.append(os.path.join(directory, f))
        self.image_paths.sort()
        self.label = label
    
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
                image = self.transform(image)

        if self.label is True:
            # Extract U value from filename
            # The regular expression `LDOS-Mu-I.*-U([0-9.]+)-.*\.png` is specifically designed for file name
            match = re.search(r'U(\d+.\d+)', os.path.basename(img_path))
            # match = re.search(r'U(\d+.)', os.path.basename(img_path))
            u_value = float(match.group(1))
            return image, torch.tensor(u_value, dtype=torch.float32)
        
        return image, torch.tensor(0.0, dtype=torch.float32)  # Dummy label if not provided
        

def get_data_loaders_regress(data_dir: str, batch_size: int=32, val_split: float=0.15,
                             test_split: float=0.15,
                             input_dim: int=256,
                             seed: int=42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates data loaders for a regression task.

    Parameters
    -----------
        data_dir: str
            Directory containing the images.
        batch_size: int
            The batch size for the data loaders.
        val_split: float
            The proportion of the dataset to use for validation.
        test_split: float
            The proportion of the dataset to use for testing.
        input_dim: int
            The dimension to resize images to.

    Returns
    ---------
        tuple: A tuple containing the training, validation, and test data loaders.
    """
    
    # 1. Image transformations
    # Initial transform without normalization
    transform = transforms.Compose([
        transforms.Resize((input_dim, input_dim)),
        transforms.ToTensor(),
    ])

    dataset = STMTBGDatasetReg(data_dir, transform=transform, label=True)
    targets = [dataset[i][1].item() for i in range(len(dataset))]

    # Get indices for each class
    class_indices = {i: [] for i in set(targets)}
    for i, label in enumerate(targets):
        class_indices[label].append(i)
    
    # train_indices, val_indices, test_indices = [], [], []
    train_indices, val_indices, test_indices = get_split_indices(len(dataset),
                                                                 val_fraction=val_split,
                                                                 test_fraction=test_split,
                                                                 seed=seed)

    # Initialize accumulators
    channels_sum = 0.0
    channels_squared_sum = 0.0
    total_pixels = 0

    for idx in train_indices:
        image, _ = dataset[idx] 
        
        # Ensure the image is in floating point format
        if not image.is_floating_point():
             image = image.float()

        # Flatten the image tensor into a single dimension of pixels
        flattened_pixels = image.view(-1)
        
        # Accumulate the counts and sums
        total_pixels += flattened_pixels.size(0)
        channels_sum += torch.sum(flattened_pixels)
        channels_squared_sum += torch.sum(flattened_pixels ** 2)

    # Calculate mean and std
    mean = channels_sum / total_pixels
    std = torch.sqrt((channels_squared_sum / total_pixels) - (mean ** 2))

    # Update the transform with normalization
    transform = transforms.Compose([
        transforms.Resize((input_dim, input_dim)),
        # HighPassFilter(4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    dataset = STMTBGDatasetReg(data_dir, transform=transform, label=True)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Class weights are not needed for regression
    return train_loader, val_loader, test_loader


def train_regress(model: torch.nn.Module,
                  train_loader: DataLoader,
                  criterion: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
    """
    Performs one epoch of training.

    Returns
    --------
        Average loss, list of predictions, and list of targets.
    """
    model.to(device)
    model.train()
    
    total_loss = 0.0
    predictions = []
    targets_list = []
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        predictions.extend(outputs.detach().cpu().numpy())
        targets_list.extend(targets.cpu().numpy())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
    
    avg_loss = total_loss / len(train_loader.dataset)    
    return avg_loss, predictions, targets_list


def evaluate_regress(model: torch.nn.Module,
                     loader: DataLoader,
                     criterion: torch.nn.Module,
                     device: torch.device) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
    """
    Performs inference over a DataLoader for validation or testing.

    Returns
    --------
        Average loss, predictions, and ground truth targets.
    """
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    predictions = []
    targets_list = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predictions.extend(outputs.detach().cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            total_loss += loss.item() * inputs.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, predictions, targets_list


def plot_stats_regress(training_stats: Dict[str, List[float]],
                       eval_stats: Dict[str, List[float]],
                       save_path: str) -> None:
    """
    Generates and saves a multi-panel plot of regression metrics over training epochs.
    """
    epochs = range(1, len(training_stats['loss']) + 1)
    
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, training_stats['loss'], label='Training Loss')
    plt.plot(epochs, eval_stats['loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, training_stats['mae'], label='Training MAE')
    plt.plot(epochs, eval_stats['mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error over Epochs')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, training_stats['mse'], label='Training MSE')
    plt.plot(epochs, eval_stats['mse'], label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error over Epochs')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, training_stats['r2'], label='Training R2')
    plt.plot(epochs, eval_stats['r2'], label='Validation R2')
    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.title('R2 Score over Epochs')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, training_stats['rmse'], label='Training RMSE')
    plt.plot(epochs, eval_stats['rmse'], label='Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f'Statistics plot saved to {save_path}')
