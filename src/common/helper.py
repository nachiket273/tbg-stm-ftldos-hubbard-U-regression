"""
helper.py
---------

Helper functions including:-
    * Load and save model weights
    * Get pretrained model
    * Get learning rate scheduler
    * Get train, test, validation split list of indices
    * Seed different libraries for reproducibility
"""
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from typing import List, Tuple, Optional, Union


def save_model(model: torch.nn.Module, model_path: str) -> None:
    """
    Safely save a PyTorch model to disk.

    The model is first written to a temporary file and then renamed to
    avoid corruption in case the process is interrupted.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model to be saved.
    model_path : str
        Destination file path.
    """
    # 1. Define a temporary path
    temp_path = model_path + ".tmp"
    
    # 2. Save the model to the temporary file
    torch.save(model.state_dict(), temp_path)
    
    # 3. Rename/move the temporary file over the final destination
    # This is typically an atomic operation on most file systems,
    # ensuring the final file is fully written and closed.
    try:
        if os.path.exists(model_path):
            os.remove(model_path) # Important to handle existing file gracefully on Windows
        os.rename(temp_path, model_path)
        print(f'Model saved safely to {model_path}')
    except Exception as e:
        print(f"Error renaming file: {e}")
        # Optionally, clean up the temp file if rename fails
        if os.path.exists(temp_path):
             os.remove(temp_path)


def load_model(model: torch.nn.Module, model_path: str, device: Optional[str]=None) -> Optional[torch.nn.Module]:
    """
    Load model weights from disk into a PyTorch model instance.

    Parameters
    ----------
    model : torch.nn.Module
        Model architecture instance into which weights will be loaded.
    model_path : str
        Path to saved model weights.
    device : str, optional
        Device on which the model should be loaded.

    Returns
    -------
    torch.nn.Module
        Model with loaded weights.
    """
    if device is None:
        device = next(model.parameters()).device
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)
    # Load the state dictionary into the model
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"RuntimeError while loading state_dict: {e}")
        return None
    model.to(device)

    # Set the model to evaluation mode
    model.eval()
    print(f'Model loaded from {model_path}')
    return model


def get_pretrained_model(model_name: str, num_classes: str,
                         num_channel: str=3, device: Optional[torch.device]=None, train: bool =False) -> torch.nn.Module:
    """
    Initializes a ResNet model (18, 34, 50, or 101) with optional pretrained weights
    and adjusts it for specific input channels and output classes.

    Parameters
    ----------
        model_name: str
            The name of the ResNet architecture (e.g., 'resnet18').
        num_classes: int
            The number of output features/classes for the final layer.
            For regression set this to 1.
        num_channel: int
            Number of input image channels (e.g., 1 for grayscale, 3 for RGB).
        device: Optional[torch.device]
            The device to load the model onto.
        train: bool
            If True, loads ImageNet pretrained weights. If False, initializes without weights.
    
    Returns
    --------
        torch.nn.model
            Modified pytorch model.

    """
    if device is None:
        device = torch.device('cpu')
    
    if train:
        if model_name == 'resnet18':
            model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        elif model_name == 'resnet34':
            model = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
        elif model_name == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        elif model_name == 'resnet101':
            model = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V1')
        else:
            raise ValueError(f"Model {model_name} not supported. Please choose from ['resnet18', 'resnet34', 'resnet50', 'resnet101']")
    else:
        if model_name == 'resnet18':
            model = models.resnet18(weights=None)
        elif model_name == 'resnet34':
            model = models.resnet34(weights=None)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=None)
        elif model_name == 'resnet101':
            model = models.resnet101(weights=None)
        else:
            raise ValueError(f"Model {model_name} not supported. Please choose from ['resnet18', 'resnet34', 'resnet50', 'resnet101']")
        
    if num_channel == 1:
        weights = model.conv1.weight.data
        model.conv1 = torch.nn.Conv2d(1, model.conv1.out_channels,
                                      kernel_size=model.conv1.kernel_size,
                                      stride=model.conv1.stride,
                                      padding=model.conv1.padding,
                                      bias=model.conv1.bias is not None)
        model.conv1.weight.data = weights.mean(dim=1, keepdim=True)

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model


def get_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str,
                     step_size: int=5, T_max: int=30) -> Union[torch.optim.lr_scheduler.StepLR,
                                                               torch.optim.lr_scheduler.ReduceLROnPlateau,
                                                               torch.optim.lr_scheduler.CosineAnnealingLR]:
    """
    Factory function to initialize a learning rate scheduler.

    Parameters
    -----------
        optimizer: torch.optim.Optimizer
            The optimizer for which to schedule the learning rate.
        scheduler_type: str
            Type of scheduler ('step', 'reduce_on_plateau', or 'cosine').
        step_size: int
            Period of learning rate decay for StepLR.
        gamma: float
            Multiplicative factor of learning rate decay.
        patience: int
            Number of epochs with no improvement before reducing LR for ReduceLROnPlateau.
        T_max: int
            Maximum number of iterations for CosineAnnealingLR.

    Returns
    --------
        _LRScheduler
            The initialized PyTorch scheduler object.
    """
    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               patience=step_size, threshold=1e-6)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    else:
        raise ValueError(f"Scheduler {scheduler_type} not supported. Please choose from ['step', 'reduce_on_plateau', 'cosine']")
    return scheduler


def get_split_indices(dataset_size: int, val_fraction: float=0.2,
                      test_fraction: float=0.1, seed: int=42) -> Tuple[List[int], List[int], List[int]]:
    """"
    Generates randomized indices for training, validation, and testing splits.

    Parameters
    -----------
        dataset_size: int
            Total number of samples in the dataset.
        val_fraction: float
            Proportion of the dataset for validation.
        test_fraction: float
            Proportion of the dataset for testing.
        seed: int
            Random seed for reproducibility.

    Returns
    ---------
        Tuple[List[int], List[int], List[int]]
            Lists containing indices for (train, val, test).
    """
    indices = list(range(dataset_size))
    if np.isclose(val_fraction, 0.0):
        train_indices, test_indices = train_test_split(indices, test_size=test_fraction, random_state=seed)
        val_indices = []
    else:    
        train_indices, temp_indices = train_test_split(indices, test_size=(val_fraction + test_fraction), random_state=seed)
        relative_test_fraction = test_fraction / (val_fraction + test_fraction)
        val_indices, test_indices = train_test_split(temp_indices, test_size=relative_test_fraction, random_state=seed)
    return train_indices, val_indices, test_indices


def seed_everything(seed: int=42) -> None:
    """
    Set random seeds for all relevant libraries to ensure reproducibility.

    This function fixes randomness for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Parameters
    ----------
    seed : int, optional
        Seed value to use for all random number generators.

    Notes
    -----
    This function should be called at the beginning of every script
    that involves training or data splitting.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GaussianNoise(torch.nn.Module):
    """Adds stochastic white noise to simulate thermal/shot noise."""
    def __init__(self, std_range=(0.005, 0.02), p=1.0):
        super().__init__()
        self.std_range = std_range
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        std = random.uniform(*self.std_range)
        return x + torch.randn_like(x) * std
    

class TipAnisotropyBlur(torch.nn.Module):
    """Simulates non-ideal tip shapes using elliptical Gaussian kernels."""
    def __init__(self, sigma_range=(0.5, 1.2), ratio_range=(1.0, 1.2), p=1.0):
        super().__init__()
        self.sigma_range = sigma_range
        self.ratio_range = ratio_range
        self.p = p

    def _get_kernel(self, sigma, ratio, theta, size=15, device='cpu'):
        ax = torch.linspace(-(size-1)/2, (size-1)/2, size, device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')

        cos_t = torch.cos(torch.tensor(theta, device=device))
        sin_t = torch.sin(torch.tensor(theta, device=device))

        x_rot = xx * cos_t - yy * sin_t
        y_rot = xx * sin_t + yy * cos_t

        kernel = torch.exp(-0.5 * (x_rot**2 / sigma**2 +
                                y_rot**2 / (sigma*ratio)**2))
        return kernel / kernel.sum()

    def forward(self, x):
        if random.random() > self.p:
            return x
        sigma = random.uniform(*self.sigma_range)
        ratio = random.uniform(*self.ratio_range)
        theta = random.uniform(0, np.pi)
        kernel = self._get_kernel(sigma, ratio, theta).to(x.device,  dtype=x.dtype)
        return F.conv2d(x.unsqueeze(0), kernel.view(1,1,15,15), padding=7).squeeze(0)
    

class SymmetryBreakingStrain(torch.nn.Module):
    """Lighter version: Simulates C1 symmetry breaking using optimized affine wrappers."""
    def __init__(self, strain_range=(0.0, 0.03), p=1.0):
        super().__init__()
        self.strain_range = strain_range
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x

        # 1. Store original type and move to Float32 for stability
        orig_dtype = x.dtype
        img = x.to(torch.float32)

        # 2. Calculate physical strain parameters
        # Strain 's' maps to scaling and shearing to break C1 symmetry
        s = random.uniform(*self.strain_range)
        
        # We translate your strain matrix logic into angle/scale/shear
        # s_angle is small, but breaks the perfect 60-degree symmetry of TBG
        scale = 1.0 + s 
        shear = [s * 10, 0] # Horizontal shear in degrees

        # 3. Apply transformation using optimized Torchvision backend
        # This avoids the F.affine_grid kernel crash issue
        img = TF.affine(
            img.unsqueeze(0), 
            angle=0, 
            translate=[0, 0], 
            scale=scale, 
            shear=shear,
            interpolation=TF.InterpolationMode.BILINEAR
        ).squeeze(0)

        # 4. Cast back to original precision (Double) if necessary for your HF data
        return img.to(orig_dtype)
    

class ScanLineNoise(torch.nn.Module):
    """Simulates correlated noise (striations) along the scan axis."""
    def __init__(self, strength_range=(0.01, 0.04), p=0.5):
        super().__init__()
        self.strength_range = strength_range
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        c, h, w = x.shape
        strength = random.uniform(*self.strength_range)
        # Noise correlated across rows (horizontal striations)
        lines = torch.randn(h, 1).repeat(1, w).to(x.device) * strength
        return x + lines


class GainShift(torch.nn.Module):
    def __init__(self, scale_range=(0.9,1.1), shift_range=(-0.02,0.02), p=0.5):
        super().__init__()
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        a = random.uniform(*self.scale_range)
        b = random.uniform(*self.shift_range)
        return a*x + b
