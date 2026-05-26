[![DOI](https://zenodo.org/badge/1161607710.svg)](https://doi.org/10.5281/zenodo.18717017)
# Prediction of Hubbard Interaction Strength from FT-LDOS Images of Moiré Systems using Machine Learning

This repository contains the code accompanying the research work:

**Prediction of Hubbard U parameter from Moiré System STM images using image recognition**

This project investigates whether interaction-dependent features encoded in simulated scanning tunneling microscopy (STM) Fourier-transformed local density of states (FT-LDOS) images can be learned using convolutional neural networks (CNNs).

The implemented framework provides dataset generation, training, evaluation, and analysis pipelines for supervised regression of interaction-dependent electronic structure features from FT-LDOS images.

---

# Scientific Motivation

Moiré systems, such as twisted bilayer graphene, exhibit strongly correlated electronic phases arising from electron–electron interactions. The Hubbard interaction strength \(U\) plays a central role in determining these phases.

Scanning tunneling microscopy (STM) provides spatially resolved measurements of the local density of states (LDOS), and its Fourier transform (FT-LDOS) reveals momentum-space information including quasiparticle interference and correlation-induced spectral features.

This work investigates whether machine learning models can learn interaction-dependent patterns in FT-LDOS images generated from theoretical simulations.

The goal is not to directly infer experimental Hubbard parameters, but to quantify and characterize interaction-dependent features encoded in STM-derived observables within a controlled theoretical framework.

---

# Repository Structure

src/ \
| - common/ \
&emsp;| - helper.py            # Utility functions: random seeding, dataset splitting, model saving/loading. \
| - fourier_transform/ \
&emsp;| - stm_ft.py            # FT-LDOS dataset generation from simulated LDOS data. \
| - regression/ \
&emsp;| - model_regress.py     # Neural network architectures: Custom CNN, Modified ResNet-18 regression model. \
&emsp;| - run_regress.py       # Training pipeline \
&emsp;| - test_regress.py      # Evaluation pipeline \
&emsp;| - visualize_regress.py # Visualization and analysis tools \
&emsp;| - grad_cam_regress.py  # Grad-CAM based model interpretability \
| - config/ \
&emsp;| - config.ini           # Hyperparameter configuration

---

# Method Overview

The pipeline consists of five stages:

1. Generate simulated FT-LDOS datasets
2. Construct augmented FT-LDOS datasets
3. Train regression models
4. Evaluate interpolation and robustness performance
5. Analyze learned representations 

The models learn a mapping: \
FT-LDOS image → interaction-dependent electronic structure features


---

# Robustness Analysis

To evaluate the robustness of the machine-learning models beyond idealized simulations, the revised version of this repository includes physically motivated FT-LDOS perturbations designed to emulate common STM imperfections.

The following augmented datasets are supported:

### Mild Perturbation Dataset
- Weak Gaussian noise
- Gain/intensity variations
- Anisotropic tip-induced blur
- Scan-line acquisition artifacts

### C3-Symmetry-Breaking Dataset
- Small affine distortions mimicking weak heterostrain
- Reduced rotational symmetry while preserving dominant FT-LDOS peaks

### Noisy Dataset
- Stronger combinations of scan-line noise, blur, gain fluctuations, and stochastic noise
- Used as a stress-test benchmark

These perturbations are used to evaluate model robustness under realistic distribution shifts.

---

# Installation

Create Python environment:

```bash
conda create -n stm_ml python=3.10
conda activate stm_ml
```

Install dependencies:

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib tqdm
pip install numba
pip install grad-cam
```

---

# Dataset Generation

Generate FT-LDOS images:

```bash
python fourier_transform/stm_ft.py --data_dir [DATA_DIR] --images_dir [IMAGES_DIR]
```
where: \
--data_dir : Base directory containing LDOS data. \
--images_dir : Target directory to save FT-LDOS images.

This produces:
* Fourier-transformed LDOS images
* Labels corresponding to interaction-dependent simulation parameters

---

# Training

Train regression model:

```bash
python regression/stm_main_regress.py --data_dir [DATA_DIR] --model_type [MODEL_TYPE] --model_path [MODEL_PATH]
```
where: \
--data_dir : Directory containing FT-LDOS images. \
--model_type :  model type - fc, conv or pretrained. \
--model_path : Path to save trained model weights. \

Features:
* Stratified dataset splitting
* Fixed random seeds
* Early stopping
* Model checkpoint saving

Supported architectures:
* Custom CNN
* ResNet-18 (modified for regression)

---

# Models

The repository contains three regression approaches:

### Linear Regression
A simple linear baseline operating directly on FT-LDOS images.

### Custom CNN
A lightweight convolutional neural network specifically designed for FT-LDOS regression.

### ResNet18 Regressor
A modified ResNet18 architecture adapted for scalar regression.

---

# Evaluation

Evaluate trained model:

```bash
python regression/test_regress.py --data_dir [DATA_DIR] --model_type [MODEL_TYPE] --model_path [MODEL_PATH]
```
where: \
--data_dir : Directory containing test set FT-LDOS images. \
--model_type :  model type - fc, conv or pretrained. \
--model_path : Path to load trained model weights. \

Evaluation metrics:
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of determination (R²)

---
# Holdout Evaluation

In addition to the standard train/validation/test split, the repository supports evaluation on a holdout dataset containing interaction strengths not present in the training set.

This benchmark is designed to assess interpolation capabilities beyond the discrete interaction values used during training.
---
# Interpretability

Guided Backpropagation: Provides insight into spatial regions contributing to gradient for backpropaation.

```bash
python regression/visualize_regress.py --data_path [DATA_PATH] --save_path [SAVE_PATH] --model_type [MODEL_TYPE] --model_path [MODEL_PATH]
```
where: \
--data_path : Directory containing FT-LDOS images. \
--save_path : Directory to save guided basckpropagation mask and images. \
--model_type :  model type - fc, conv or pretrained. \
--model_path : Path to load trained model weights. \

Grad-CAM visualization: Provides insight into spatial regions contributing to model predictions.

```bash
python regression/grad_cam_regress.py --data_path [DATA_PATH] --save_path [SAVE_PATH] --model_type [MODEL_TYPE] --model_path [MODEL_PATH]
```
where: \
--data_path : Directory containing FT-LDOS images. \
--save_path : Directory to save grad-cam masked images. \
--model_type :  model type - fc, conv or pretrained. \
--model_path : Path to load trained model weights. \

---

# Configuration

Hyperparameters are defined in:
```bash
src/config/config.ini
```
including:
* Number of k-points
* Initial Learning Rate
* Number of Epochs
* Batch Size
* Dropout Rate
* Learning Rate Scheduler
* Early Stoppin Patience
* Random Seed
* and Others...

---

# Reproducibility

All results reported in the revised manuscript can be reproduced using the scripts contained in this repository.

The repository includes:

- FT-LDOS generation
- Training pipelines
- Robustness evaluation
- Holdout-set analysis
- PCA analysis
- Guided backpropagation
- Grad-CAM visualization

Random seeds used in the manuscript are provided in the configuration files.

---

# Intended Use

This repository is intended for:

* Computational condensed matter physics research
* Machine learning analysis of simulated STM data
* Methodological studies of feature extraction from electronic structure simulations

This repository is not intended for direct experimental parameter inference without additional calibration and validation on experimental STM datasets.

The present implementation should be viewed as a proof-of-principle framework for studying interaction-dependent features in theoretically generated FT-LDOS images.

---

# Citation

```
@misc{tanksale2026predictionatomistichubbardu,
      title={Prediction of the atomistic Hubbard U interaction from moir\'e system STM-images using image recognition}, 
      author={Nachiket Tanksale and Tobias Stauber},
      year={2026},
      eprint={2602.18890},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mes-hall},
      url={https://arxiv.org/abs/2602.18890}, 
}
```
