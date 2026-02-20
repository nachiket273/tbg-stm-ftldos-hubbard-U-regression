"""
stm_ft.py
---------

Dataset generation script for producing Fourier-transformed local density of states (FT-LDOS)
images from simulated real-space LDOS data.

Scientific purpose
------------------
In scanning tunneling microscopy (STM), the local density of states (LDOS) provides spatially
resolved information about the electronic structure of a system. Fourier transforming LDOS
images into momentum space produces FT-LDOS, which reveals characteristic wavevectors such
as moiré Bragg peaks and quasiparticle interference patterns.

These FT-LDOS images serve as input features for supervised machine learning models that learn
interaction-dependent electronic structure features.

This script performs the following operations:

1. Loads simulated LDOS data
2. Interpolates irregular spatial grids onto uniform grids
3. Computes Fourier transforms of LDOS images
4. Stores real-space and momentum-space images
5. Organizes outputs into structured 

Usage
-------
python fourier_transform/stm_ft.py --data_dir [DATA_DIR] --images_dir [IMAGES_DIR]

Outputs
-------

Directory structure:

data/
    imgs/
        ldos/
            complete/
        kspace/
            complete/

These outputs are later used for training neural network regression models.

Reproducibility
---------------
All parameters are controlled via configuration files and fixed seeds when applicable.
"""
import argparse
import configparser
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
import os
from scipy.interpolate import griddata
from typing import Dict, Tuple, Optional, Any


def create_directories(base_data_dir: str) -> Dict[str, str]:
    """
    Creates the necessary output directory structure for images.

    Parameters
    ------------
        base_data_dir: str
            String path to the base directory.

    Returns
    ----------
        Dict[str, str]
            dictionary containing path to real space image directory
            and FT-LDOS image directory.
    """
    imgs_dir = os.path.join(base_data_dir, 'imgs')
    
    # LDOS real-space image directories
    ldos_dir = os.path.join(imgs_dir, 'ldos')
    ldos_complete_dir = os.path.join(ldos_dir, 'complete')

    # K-space (Fourier Transform) image directories
    kspace_dir = os.path.join(imgs_dir, 'kspace')
    kspace_complete_dir = os.path.join(kspace_dir, 'complete')

    # Create all directories
    os.makedirs(ldos_complete_dir, exist_ok=True)
    os.makedirs(kspace_complete_dir, exist_ok=True)

    return {
        'ldos_complete': ldos_complete_dir,
        'kspace_complete': kspace_complete_dir
    }


def plot_ldos_real_space(x: np.ndarray, y: np.ndarray, ldos_values: np.ndarray,
                         filename: str, output_dir: str, grid_size: int=501) -> None:
    """
    Plots the real-space LDOS data using interpolation and saves it as a clean image.
    No labels, legends, or color bars.

    Parameters
    -----------
        x: np.ndarray
            X-coordinates of the data points.
        y: np.ndarray
            Y-coordinates of the data points.
        ldos_values: np.ndarray
            The intensity value at each (x, y) point.
        filename: str
            Name of the output file.
        output_dir: str
            Directory where the image will be saved.
        grid_size: int
            Resolution of the interpolation grid.
            This value is read from config file.
            Defaults to 501.

    """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    grid_x, grid_y = np.mgrid[x_min:x_max:grid_size*1j, y_min:y_max:grid_size*1j]
    grid_ldos = griddata((x, y), ldos_values, (grid_x, grid_y), method='cubic')
    grid_ldos = np.nan_to_num(grid_ldos, nan=0.0)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_ldos.T, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='coolwarm')
    plt.axis('off') # Turn off axes
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0, dpi=64) # Save without padding/borders
    plt.close()


@njit(parallel=True)
def calculate_fourier_transform_numba(r_coords: np.ndarray, ldos_values: np.ndarray,
                                      num_k_points: int=201) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculates the Fourier transform of LDOS using the required summation formula,
    optimized and parallelized using Numba.

    Parameters
    -----------
        r_coords: np.ndarray
            Array of shape (N, 2) containing real-space coordinates.
        ldos_values: np.ndarray
            Array containing intensity values at those coordinates.
        num_k_points: int
            The number of k-points in k-space.
    
    Returns
    --------
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: 
            The complex FT array and the k_x and k_y range arrays.
    """
    if r_coords.shape[0] == 0:
        # Handle empty data
        size = 2 * num_k_points + 1
        return np.zeros((size, size), dtype=np.complex128), None, None

    # Setup k-space ranges
    # Note: These ranges are slightly different from your original, but maintain the structure
    # Numba requires explicit float operations
    N = 2 * num_k_points + 1
    
    # Calculate the max k extent (This must be correct based on the lattice constant of the sample)
    # The original scaling factors are preserved: 2*pi * 1.1 / N and 4*pi * 1.1 / (N * sqrt(3))
    k_x_range =  np.linspace(-1.1, 1.1, N) * 2 * np.pi 
    k_y_range =  np.linspace(-1.1, 1.1, N) * 4 * np.pi / np.sqrt(3)


    # The array initialization must be done outside the parallel loop
    transformed_ldos = np.zeros((len(k_x_range), len(k_y_range)), dtype=np.complex128)

    # Use prange for parallel execution over the k-space grid (the heavy loop)
    # The outer loop is parallelized
    for i in prange(len(k_x_range)): 
        k_x = k_x_range[i]
        
        # The inner loop remains sequential
        for j in range(len(k_y_range)):
            k_y = k_y_range[j]
            
            # Vector operations (k_dot_r and np.sum) are still highly optimized
            k_dot_r = k_x * r_coords[:, 0] + k_y * r_coords[:, 1]
            
            # Note: We must use the NumPy implementation of exp since Numba's is complex
            # Numba is smart enough to handle np.exp within its njit scope.
            transformed_ldos[i, j] = np.sum(ldos_values * np.exp(-1j * k_dot_r))

    return transformed_ldos, k_x_range, k_y_range


def plot_fourier_transform(ft_data: np.ndarray, k_x_range: Optional[np.ndarray],
                           k_y_range: Optional[np.ndarray], filename: str, output_dir: str) -> None:
    """
    Plots the Fourier transform data (magnitude, real, or imaginary) and saves it as a clean image.
    No labels, legends, or color bars.

    Parameters
    -----------
        ft_data: np.ndarray
            Complex Fourier transform data.
        k_x_range: Optional[np.ndarray]
            The range of k_x values used.
        k_y_range: Optional[np.ndarray]
            The range of k_y values used.
        filename: str
            Name of the output file.
        output_dir: str
            Directory where the image will be saved.
    """
    if k_x_range is None or k_y_range is None:
        print(f"Skipping plot for {filename} due to invalid k-space ranges.")
        return

    plt.figure(figsize=(8, 8))
    plt.imshow(np.abs(ft_data).T, extent=(k_x_range.min(), k_x_range.max(), k_y_range.min(), k_y_range.max()),
               cmap='turbo', origin='lower', aspect='auto')
    plt.axis('off') # Turn off axes
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0, dpi=112) # Save without padding/borders
    plt.close()


# --- Main processing logic ---
if __name__ == '__main__':
    # Argument parser for command line options
    parser = argparse.ArgumentParser(description='Process LDOS data and generate foruier transformed images.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data', help='Base directory for data processing.',)
    parser.add_argument('--images_dir', type=str, default='data', help='Directory to save output images.')

    # Define the base 'data' directory (one level up from 'data/data')
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.images_dir
    input_ldos_dir = os.path.join(data_dir, 'data', 'ldos')

    # Load configuration
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    grid_size = config.getint('Fourier Transform', 'grid_size', fallback=501)
    num_k_points = config.getint('Fourier Transform', 'num_k_points', fallback=201)
    print(f"Using grid size: {grid_size}, number of k-points: {num_k_points}")

    # Create output directories
    output_dirs = create_directories(output_dir)

    # Process each LDOS file
    if not os.path.exists(input_ldos_dir):
        print(f"Error: Input directory '{input_ldos_dir}' not found. Please ensure your 'data/data/ldos' structure is correct.")
    else:
        for filename in os.listdir(input_ldos_dir):
            if filename.endswith('.dat'):
                file_path = os.path.join(input_ldos_dir, filename)
                print(f"Processing {filename}...")

                # Load complete LDOS data
                complete_data = np.loadtxt(file_path)
                r_coords_complete = complete_data[:, :2]
                ldos_values_complete = complete_data[:, 2]

                # Generate base filename for outputs (e.g., "LDOS-Mu-I9-...")
                base_name = os.path.splitext(filename)[0]

                # --- Plotting Real-Space LDOS ---
                # Complete LDOS
                plot_ldos_real_space(r_coords_complete[:, 0], r_coords_complete[:, 1], ldos_values_complete,
                                      f'{base_name}_complete_ldos.png',
                                      output_dirs['ldos_complete'], grid_size=grid_size)

                # --- Fourier Transform and Plotting ---
                # Complete LDOS FT
                ft_complete, k_x_range_c, k_y_range_c = calculate_fourier_transform_numba(r_coords_complete, ldos_values_complete,
                                                                                          num_k_points=num_k_points)
                plot_fourier_transform(ft_complete, k_x_range_c, k_y_range_c,
                                       f'{base_name}_ft_magnitude.png', output_dirs['kspace_complete'])

        print("Processing complete. Images saved to the 'data/imgs' directory.")
