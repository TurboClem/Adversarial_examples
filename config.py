"""
Configuration file for the EuroSat adversarial robustness project
"""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset configuration
DATA_PATH_TRAIN = "datasets/EuroSAT_RGB/train_clean"
DATA_PATH_EVAL = "datasets/EuroSAT_RGB/test_clean"
NUM_CLASSES = 10
IMG_SIZE = 64  # Resize images to 64x64 for faster training

# Training configuration
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = int(EPOCHS / 5)
SEED = 42  # For reproducibility
MADRY = None

# Model configuration
MODEL_TYPE = "simple_cnn"  # Options: 'simple_cnn', 'resnet'

# Paths
SAVE_MODEL_PATH = "outputs/models/"
SAVE_PLOTS_PATH = "outputs/plots/"

# Data transforms
MEAN = [0.3446759581565857, 0.3805992603302002, 0.40795156359672546]  # Computed on train set
STD = [0.09143105894327164, 0.06515809893608093, 0.05527787283062935]  # Computed on train set
