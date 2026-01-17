# EuroSat Adversarial Robustness Project

## 📋 Project Overview
This project studies adversarial robustness of image classification models on satellite imagery (EuroSat dataset). The goal is to implement and compare different CNN architectures, then analyze their vulnerability to adversarial attacks.

## 🏗️ Project Structure
```
Adversarial_examples/
├── main.py                    # Main entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── download_data.sh           # Downloads EuroSat dataset
├── README.md                  # This file
├── data/EuroSAT_RGB/          # EuroSat dataset when downloaded
│   ├── AnnualCrop/
│   ├── Forest/
│   ├── HerbaceousVegetation/
│   ├── Highway/
│   ├── Industrial/
│   ├── Pasture/
│   ├── PermanentCrop/
│   ├── Residential/
│   ├── River/
│   └── SeaLake/
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── simple_cnn.py         # Custom CNN implementation
│   ├── resnet.py             # ResNet implementation
│   └── resnet_advprop.py     # ResNet Advprop implementation
├── data_loader/              # Data handling
│   ├── __init__.py
│   └── dataset.py            # EuroSat dataset class
├── train/                    # Training utilities
│   ├── __init__.py
│   ├── trainer.py            # Model trainer class
│   └── advprop_trainer.py    # Model trainer for advprop class
├── utils/                    # Helper functions
│   ├── evaluation_logger.py
│   ├── result_collector.py
│   ├── utils.py
│   └── visualization.py      # Plotting utilities
└── outputs/                  # Generated outputs
    ├── evaluation_logs/      # Saved evaluation results
    ├── models/               # Saved model checkpoints
    └── plots/                # Training plots
```

## 🚀 Initialisation and launching the project

### 0. Install the libraries
```bash
pip install -r requirements.txt
```

### 1. Load datasets
```bash
# Download data to workspace (change the workspace if you do not use sspcloud)
bash download_data.sh
```

### 2. Prepare Datasets
```bash
# The dataset should already be in:
# data/EuroSAT_RGB/ (27000 images, 10 classes)

# Verify dataset structure
ls -la data/EuroSAT_RGB/
# Should show 10 folders: AnnualCrop, Forest, HerbaceousVegetation, etc.
```

- Run the notebook *create_datasets.ipynb* to create the reproductible clean train and test sets, train teh baseline model, and create the attacked test set.

### 3. Run the Project

To follow our framework:
    - Run the notebook ** to reproduce the experiment on Ilyas et al.
    - Run the notebook ** to reproduce the experiment on Madry et al.
    - Run the notebook ** to reproduce the experiment on Xie et al.

## How to run a training

### 📊 Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--model` | Choose model architecture | `--model simple_cnn` |
| `--seed` | Seed to reproduce| `--seed 42` |
| `--epochs` | Number of training epochs | `--epochs 20` |
| `--batch-size` | Batch size for training | `--batch-size 64` |
| `--lr` | Learning rate | `--lr 0.0001` |
| `--patience` | Early stopping | `--patience 10` |
| `--data-path-train` | Path to train dataset | `--data-path-train datasetsEuroSAT_RGB/train_clean` |
| `--data-path-eval` | Path to test dataset | `--data-path-eval datasetsEuroSAT_RGB/test_clean` |
| `--train` | Train the model | `--train` |
| `--evaluate` | Evaluate on test set | `--evaluate` |
| `--visualize` | Visualize predictions | `--visualize` |
| `--save-model-path` | Path to save the model | `--save-model-path outputs/model` |
| `--save-plots-path` | Path to save plots | `--save-plots-path outputs/plots` |
| `--advprop` | Use the Advprop procedure during training | `--advprop` |
| `--epsilon` | Epsilon for PGD during an Advprop training | `--epsilon 0.2` |
| `--advprop-iterations` | Number of iterations for PGD during an Advprop training | `--advprop-iterations 10` |
| `--madry` | To use a minmax optimisation training | `--madry None` |


### 💻 Example Workflows

#### **Workflow 1: Full Training Pipeline**
```bash
# Step 1: Train Simple CNN on RGB images
python main.py --model simple_cnn --train --epochs 15 --data-path-train datasets/EuroSAT_RGB/train_clean

# Step 2: Evaluate and visualize
python main.py --model simple_cnn --evaluate --visualize --data-path-eval datasets/EuroSAT_RGB/test_clean --save-model-path outputs/model --save-plots-path outputs/plots
```

#### **Workflow 2: Quick Evaluation**
```bash
# Just evaluate an existing model
python main.py --model simple_cnn --evaluate --data-path-eval data/EuroSAT_RGB/test_clean --save-plots-path outputs/plots --save-model-path outputs/model
```

### 🛠️ Configuration
Edit `config.py` to modify:
- Image size (default: 64x64)
- Batch size (default: 32)
- Learning rate (default: 0.001)
- Data augmentation
- Model save paths

**Important**: Update `DATA_PATH` in `config.py` to match your dataset location:
```python
DATA_PATH = 'data/EuroSAT_RGB'  # For RGB images
# or
DATA_PATH = 'data/EuroSAT_MS'   # For multi-spectral (advanced)
```

### 📈 Outputs
The script automatically creates:
- **Model checkpoints** in `outputs/models/`
- **Training plots** in `outputs/plots/`
- **Console logs** of training progress

## 🎯 Dataset Information

### **EuroSAT_RGB (Recommended for your project):**
- **Images**: 27,000
- **Classes**: 10
- **Size**: 64x64 pixels
- **Format**: RGB (3 channels)
- **Classes**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake


## 🐛 Troubleshooting

### **Issue: "No such file or directory: 'data/EuroSAT_RGB'"**
```bash
# Check what's in your data folder
ls -la data/

# If you have different structure, update config.py
# Or use --data-path argument
python main.py --train --data-path ./data/EuroSAT_RGB
```

### **Issue: CUDA Out of Memory (SSPCloud GPU limits)**
```bash
# Reduce batch size
python main.py --batch-size 16

# Reduce image size (edit config.py)
# Change IMG_SIZE = 32
```

### **Issue: Import errors on SSPCloud**
```bash
# Make sure you're in the project directory
cd ~/work/Adversarial_examples

# Install dependencies
pip install -r requirements.txt
```
---