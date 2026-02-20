"""
stm_main_regress.py
-------------------
The primary entry point for training and validating STM regression models.

This script orchestrates the entire machine learning pipeline:
1. Loads hyperparameters and execution settings from config files.
2. Initializes data loaders for training, validation, and testing.
3. Sets up the model architecture (FC, Conv, or ResNet-based).
4. Executes the training loop for a specified number of epochs.
5. Monitors performance using metrics (MAE, MSE, R2, RMSE).
6. Saves the best model weights and generates training history plots.

Usage:
    python regression/stm_main_regress.py --data_dir [DATA_DIR] --model_type [MODEL_TYPE] --model_path [MODEL_PATH]
"""
import argparse
import configparser
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from common.helper import save_model, load_model, seed_everything
from common.helper import get_lr_scheduler, get_pretrained_model
from regression.run_regress import get_data_loaders_regress, plot_stats_regress
from regression.run_regress import train_regress, evaluate_regress
from regression.model_regress import SimpleFCRegressorModel, SimpleConvRegressorModel, initialize_weights, HybridResNetRegressor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a regression model on images.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset.')
    parser.add_argument('--model_type', type=str, choices=['fc', 'conv', 'pretrained'], default='conv',
                        help='Type of model to use: "fc" for fully connected, "conv" for convolutional, "pretrained" for pretrained model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--train_on_combined_ds', action="store_true",
                        help='Train on combined dataset(train + val) for final evaluation')
 
    args = parser.parse_args()

    # Read parameters from config file
    config = configparser.ConfigParser()
    config_path = os.path.join(project_root, 'config', 'config.ini')
    config.read(config_path)

    input_dim = config.getint('Common Parameters', 'input_size', fallback=256)
    input_channels = config.getint('Common Parameters', 'input_channels', fallback=3)
    device_type = config.get('Common Parameters', 'device', fallback='cpu')
    seed = config.getint('Common Parameters', 'random_seed', fallback=42)
    start_lr = config.getfloat('Common Parameters', 'learning_rate', fallback=0.001)
    batch_size = config.getint('Common Parameters', 'batch_size', fallback=64)
    val_split = config.getfloat('Common Parameters', 'validation_split', fallback=0.2)
    test_split = config.getfloat('Common Parameters', 'test_split', fallback=0.2)
    epochs = config.getint('Common Parameters', 'num_epochs', fallback=30)
    dropout_rate = config.getfloat('Common Parameters', 'dropout_rate', fallback=0.5)
    weight_decay = config.getfloat('Common Parameters', 'weight_decay', fallback=1e-4)
    early_stopping_patience = config.getint('Common Parameters', 'early_stopping_patience', fallback=7)
    lr_scheduler = config.get('Common Parameters', 'lr_scheduler', fallback='step')
    model_name = config.get('Common Parameters', 'model', fallback='resnet18')

    loss_name = config.get('Regression', 'loss_function', fallback='mean_squared_error')

    if val_split + test_split >= 1.0:
        raise ValueError("The sum of validation_split and test_split must be less than 1.0")
    if val_split < 0.0 or test_split < 0.0:
        raise ValueError("validation_split and test_split must be non-negative")
    if val_split == 0.0 :
        raise ValueError("Validation_split must be greater than 0.0")
    if test_split == 0.0 :
        raise ValueError("Test_split must be greater than 0.0")

    # Seed all randomness
    seed_everything(seed)

    device = torch.device(device_type)
    print(f'Using device: {device}')

    # Get parent directory of model path for saving plots
    model_dir = os.path.dirname(args.model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # save stats
    train_stats = {'loss': [], 'mae': [], 'mse': [], 'r2': [], 'rmse': [] }
    val_stats = {'loss': [], 'mae': [], 'mse': [], 'r2': [], 'rmse': [] }
    test_stats = {'loss': float('inf'), 'mae': 0.0, 'mse': 0.0, 'r2': 0.0, 'rmse': 0.0 }

    # Load data
    train_loader, val_loader, test_loader = get_data_loaders_regress(
        args.data_dir, 
        batch_size=batch_size, 
        val_split=val_split, 
        test_split=test_split, 
        input_dim=input_dim,
        seed=seed
    )

    # Save mean and std for reference from train_loader transform to use during inference
    # in config file
    mean = train_loader.dataset.dataset.transform.__dict__['transforms'][-1].mean
    std = train_loader.dataset.dataset.transform.__dict__['transforms'][-1].std
    
    config.set('Common Parameters', 'mean', str(mean))
    config.set('Common Parameters', 'std', str(std))

    # Write the updated config back to file
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    # Initialize model
    if args.model_type == 'fc':
        model = SimpleFCRegressorModel(input_dim=input_dim)
        initialize_weights(model)
    elif args.model_type == 'conv':
        model = SimpleConvRegressorModel(input_channels=input_channels,
                                         dropout_rate=dropout_rate)
        initialize_weights(model)
    else:  # pretrained
        model = get_pretrained_model(model_name, 1, num_channel=input_channels,
                                     device=device, train=True)
        model = HybridResNetRegressor(model)
        
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)

    if loss_name == 'mean_squared_error':
        criterion = nn.MSELoss()
    elif loss_name == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.L1Loss()

    best_val_loss = float('inf')
    scheduler = get_lr_scheduler(optimizer, lr_scheduler, T_max=epochs)

    # Early stopping parameters
    no_improve_epochs = 0

    for epoch in range(epochs):
        train_loss, train_preds, train_tgts = train_regress(model, train_loader, criterion, optimizer, device)
        train_stats['loss'].append(train_loss)
        mae = mean_absolute_error(train_tgts, train_preds)
        mse = mean_squared_error(train_tgts, train_preds)
        r2 = r2_score(train_tgts, train_preds)
        rmse = root_mean_squared_error(train_tgts, train_preds)
        train_stats['mae'].append(mae)
        train_stats['mse'].append(mse)
        train_stats['r2'].append(r2)
        train_stats['rmse'].append(rmse)
        val_loss, val_preds, val_tgts = evaluate_regress(model, val_loader, criterion, device)
        val_stats['loss'].append(val_loss)
        mae = mean_absolute_error(val_tgts, val_preds)
        mse = mean_squared_error(val_tgts, val_preds)
        r2 = r2_score(val_tgts, val_preds)
        rmse = root_mean_squared_error(val_tgts, val_preds)
        val_stats['mae'].append(mae)
        val_stats['mse'].append(mse)
        val_stats['r2'].append(r2)
        val_stats['rmse'].append(rmse)
        if lr_scheduler == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f'Epoch {epoch+1}/{epochs} - '
              f'\nTrain Loss: {train_loss:.4f}, MAE: {train_stats["mae"][-1]:.4f}, MSE: {train_stats["mse"][-1]:.4f}, R2: {train_stats["r2"][-1]:.4f}, RMSE: {train_stats["rmse"][-1]:.4f}'
                f'\nVal Loss: {val_loss:.4f}, MAE: {val_stats["mae"][-1]:.4f}, MSE: {val_stats["mse"][-1]:.4f}, R2: {val_stats["r2"][-1]:.4f}, RMSE: {val_stats["rmse"][-1]:.4f}')

        if val_loss < best_val_loss:
            no_improve_epochs = 0
            best_val_loss = val_loss
            save_model(model, args.model_path)
            print(f'Best model saved at location: {args.model_path}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f'Early stopping triggered after {early_stopping_patience} epochs with no improvement.')
                break
            
    # Plot stats
    save_path = os.path.join(model_dir, 'training_stats.png') if model_dir else 'training_stats.png'
    plot_stats_regress(train_stats, val_stats, save_path)

    # Now train on combined train + val set
    if args.train_on_combined_ds:
        combined_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        # Re-Initialize model and optimizer
        # Initialize model
        if args.model_type == 'fc':
            model = SimpleFCRegressorModel(input_dim=input_dim)
            initialize_weights(model)
        elif args.model_type == 'conv':
            model = SimpleConvRegressorModel(input_channels=input_channels,
                                            dropout_rate=dropout_rate)
            initialize_weights(model)
        else:  # pretrained
            model = get_pretrained_model(model_name, 1, num_channel=input_channels, device=device, train=True)
            model = HybridResNetRegressor(model)
        
        for param in model.parameters():
            param.requires_grad = True
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)

        if loss_name == 'mean_squared_error':
            criterion = nn.MSELoss()
        elif loss_name == 'huber':
            criterion = nn.HuberLoss(delta=1.0)
        else:
            criterion = nn.L1Loss()

        best_loss = float('inf')
        scheduler = get_lr_scheduler(optimizer, lr_scheduler, T_max=epochs)

        no_improve_epochs = 0

        for epoch in range(epochs):
            train_loss, train_preds, train_tgts = train_regress(model, combined_loader, criterion, optimizer, device)
            mae = mean_absolute_error(train_tgts, train_preds)
            mse = mean_squared_error(train_tgts, train_preds)
            r2 = r2_score(train_tgts, train_preds)
            rmse = root_mean_squared_error(train_tgts, train_preds)

            print(f'[Combined Train + Val] Epoch {epoch+1}/{epochs} - '
                f'\nTrain Loss: {train_loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}')
            
            if train_loss < best_loss:
                no_improve_epochs = 0
                best_loss = train_loss
                save_model(model, args.model_path)
                print(f'Best model saved at location: {args.model_path}')
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_patience:
                    print(f'Early stopping triggered after {early_stopping_patience} epochs with no improvement.')
                    break

            if lr_scheduler == 'reduce_on_plateau':
                scheduler.step(train_loss)
            else:
                scheduler.step()


    # Load best model for testing
    if args.model_type == 'fc':
        model = SimpleFCRegressorModel(input_dim=input_dim)
    elif args.model_type == 'conv':
        model = SimpleConvRegressorModel(input_channels=input_channels,
                                         dropout_rate=dropout_rate)
    else:  # pretrained
        model = get_pretrained_model(model_name, 1, num_channel=input_channels, device=device, train=False)
        model = HybridResNetRegressor(model)
        
    model = load_model(model, args.model_path)
    model.to(device)
    model.eval()

    if loss_name == 'mean_squared_error':
        criterion = nn.MSELoss()
    elif loss_name == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.L1Loss()

    test_loss, test_preds, test_tgts = evaluate_regress(model, test_loader, criterion, device)
    mae = mean_absolute_error(test_tgts, test_preds)
    mse = mean_squared_error(test_tgts, test_preds)
    r2 = r2_score(test_tgts, test_preds)
    rmse = root_mean_squared_error(test_tgts, test_preds)
    test_stats['loss'] = test_loss
    test_stats['mae'] = mae
    test_stats['mse'] = mse
    test_stats['r2'] = r2
    test_stats['rmse'] = rmse

    print(f'Test Loss: {test_loss:.4f}, MAE: {test_stats["mae"]:.4f}, MSE: {test_stats["mse"]:.4f}, R2: {test_stats["r2"]:.4f}, RMSE: {test_stats["rmse"]:.4f}')
    for tgt, pred in zip(test_tgts, test_preds):
        print(f'Target: {tgt.tolist()[0]:.4f}, Prediction: {pred.tolist()[0]:.4f}')
