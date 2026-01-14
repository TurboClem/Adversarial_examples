"""
Configuration file for the EuroSat adversarial robustness project
"""
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configuration
DATA_PATH_TRAIN = 'datasets/EuroSAT_RGB/train_clean'
DATA_PATH_EVAL = 'datasets/EuroSAT_RGB/test_clean'
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
MODEL_TYPE = 'simple_cnn'  # Options: 'simple_cnn', 'resnet'

# Paths
SAVE_MODEL_PATH = 'outputs/models/'
SAVE_PLOTS_PATH = 'outputs/plots/'

# Data transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]