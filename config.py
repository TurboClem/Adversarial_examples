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
<<<<<<< HEAD
MEAN = [0.34467601776123047, 0.38059931993484497, 0.40795138478279114]  # Computed on train set
STD = [0.09143105149269104, 0.06515813618898392, 0.05527789145708084]  # Computed on train set
=======
MEAN = [0.34467610716819763, 0.38059931993484497, 0.40795162320137024]  # Computed on train set
STD = [0.09143105149269104, 0.06515812873840332, 0.055277884006500244]  # Computed on train set
>>>>>>> b3f4b2adb7cb97cd57aab7979b91c455836a4585
