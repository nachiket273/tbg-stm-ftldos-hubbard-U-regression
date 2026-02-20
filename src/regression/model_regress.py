"""
model_regress.py
----------------
Defines neural network architectures for regression tasks on STM/FT-LDOS data.

This module provides:
1. Weight initialization utilities.
2. SimpleFCRegressorModel: A baseline fully connected network.
3. SimpleConvRegressorModel: A custom CNN with Batch Normalization and Dropout.
4. HybridResNetRegressor: A wrapper for using ResNet backbones in regression.

The models are designed to map 2D spectroscopic images to physical scalar values 
(e.g., interaction strength 'U').
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(model: nn.Module) -> None:
    """
    Initializes model weights using Kaiming Normal (He) initialization for 
    layers with ReLU activations and constant initialization for Batch Norm.

    Parameters
    -----------
        model: nn.Module
            The PyTorch module to initialize.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)


class SimpleFCRegressorModel(nn.Module):
    """
    A baseline Fully Connected (MLP) Regressor.
    Flattens the 2D input into a 1D vector before processing.
    """
    def __init__(self, input_dim: int):
        """
        Parameters
        -----------
            input_dim: int
                The height/width of the square input image.
        """
        super(SimpleFCRegressorModel, self).__init__()
        self.input_features = input_dim * input_dim 
        self.fc1 = nn.Linear(self.input_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input tensor
        x = x.view(-1, self.input_features)

        # Apply the fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class SimpleConvRegressorModel(nn.Module):
    """
    A custom Convolutional Neural Network designed for STM images.
    Uses multiple Conv-ReLU-BatchNorm blocks followed by Global Average Pooling.
    """
    def __init__(self, input_channels: int=3, dropout_rate: float=0.5):
        """
        Parameters
        -----------
            input_channels: int
                Number of input channels (e.g., 1 for grayscale).
            dropout_rate: float
                Probability of setting neurons to zero during training.
        """
        super(SimpleConvRegressorModel, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, dim=1)
        x = self.classifier(x)
        x = torch.clamp(x, min=0.0)
        return x


class HybridResNetRegressor(nn.Module):
    """
    A wrapper that adapts a ResNet backbone for regression.
    The original ResNet architecture is sliced to remove the final classification head.
    """
    def __init__(self, base_model: nn.Module):
        """
        Paramters
        ----------
            base_model: nn.Module
                A torchvision ResNet model (e.g., ResNet18).
        """
        super().__init__()
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])
        self.fc_final = nn.Linear(base_model.fc.in_features, 1)
    
    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        img_feat = self.cnn(x_img).flatten(1)
        img_feat = F.normalize(img_feat, dim=1)
        img_feat = self.fc_final(img_feat)
        img_feat = torch.clamp(img_feat, min=0.0)
        return img_feat
