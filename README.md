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
├── data/                      # EuroSat dataset (2 versions)
│   ├── EuroSAT_RGB/           # RGB version (we use this one)
│   │   ├── AnnualCrop/
│   │   ├── Forest/
│   │   ├── HerbaceousVegetation/
│   │   ├── Highway/
│   │   ├── Industrial/
│   │   ├── Pasture/
│   │   ├── PermanentCrop/
│   │   ├── Residential/
│   │   ├── River/
│   │   └── SeaLake/
│   └── EuroSAT_MS/            # Multi-spectral version (ignore for now)
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── simple_cnn.py         # Custom CNN implementation
│   └── resnet.py             # ResNet implementation
├── data_loader/              # Data handling
│   ├── __init__.py
│   └── dataset.py            # EuroSat dataset class
├── train/                    # Training utilities
│   ├── __init__.py
│   └── trainer.py           # Model trainer class
├── utils/                    # Helper functions
│   ├── __init__.py
│   └── visualization.py     # Plotting utilities
└── outputs/                  # Generated outputs
    ├── models/              # Saved model checkpoints
    └── plots/               # Training plots
```

## 🚀 Quick Start

### 1. Load datasets
```bash
# Download data to workspace (change the workspace if you do not use sspcloud)
bash download_data.sh
```

### 2. Prepare Dataset
```bash
# The dataset should already be in:
# data/EuroSAT_RGB/ (27000 images, 10 classes)
# data/EuroSAT_MS/ (multi-spectral version - ignore for now)

# Verify dataset structure
ls -la data/EuroSAT_RGB/
# Should show 10 folders: AnnualCrop, Forest, HerbaceousVegetation, etc.
```

### 3. Configure the Project
Edit `config.py` to set the correct data path:
```python
# Change this line in config.py
DATA_PATH = 'data/EuroSAT_RGB'  # Use RGB version
```

### 4. Run the Project

#### **Train Simple CNN Model:**
```bash
python main.py --model simple_cnn --train --epochs 15
```

#### **Train ResNet18 Model:**
```bash
python main.py --model resnet18 --train --epochs 20
```

#### **Evaluate a Trained Model:**
```bash
# Evaluate and visualize predictions
python main.py --model simple_cnn --evaluate --visualize
```

#### **All Options:**
```bash
# Show all available options
python main.py --help

# Expected output:
# usage: main.py [-h] [--model {simple_cnn,resnet18}] [--epochs EPOCHS]
#                [--batch-size BATCH_SIZE] [--lr LR] [--data-path DATA_PATH]
#                [--train] [--evaluate] [--visualize]
```

## 📊 Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--model` | Choose model architecture | `--model simple_cnn` |
| `--epochs` | Number of training epochs | `--epochs 20` |
| `--batch-size` | Batch size for training | `--batch-size 64` |
| `--lr` | Learning rate | `--lr 0.0001` |
| `--data-path` | Path to dataset | `--data-path data/EuroSAT_RGB` |
| `--train` | Train the model | `--train` |
| `--evaluate` | Evaluate on test set | `--evaluate` |
| `--visualize` | Visualize predictions | `--visualize` |

## 💻 Example Workflows

### **Workflow 1: Full Training Pipeline**
```bash
# Step 1: Train Simple CNN on RGB images
python main.py --model simple_cnn --train --epochs 15 --data-path data/EuroSAT_RGB

# Step 2: Evaluate and visualize
python main.py --model simple_cnn --evaluate --visualize

# Step 3: Train ResNet for comparison
python main.py --model resnet18 --train --epochs 20
```

### **Workflow 2: Quick Evaluation**
```bash
# Just evaluate an existing model
python main.py --model simple_cnn --evaluate --data-path data/EuroSAT_RGB
```

### **Workflow 3: Compare RGB vs MS (Advanced)**
```bash
# Train on RGB version
python main.py --model simple_cnn --train --epochs 15 --data-path data/EuroSAT_RGB

# Train on MS version (for future multi-spectral analysis)
python main.py --model simple_cnn --train --epochs 15 --data-path data/EuroSAT_MS
```

## 🛠️ Configuration
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

## 📈 Outputs
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

### **EuroSAT_MS (Multi-spectral):**
- 13 spectral bands
- More complex, for advanced analysis
- Use RGB version for your adversarial robustness experiments

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

## 🧪 Testing Your Setup
```bash
# Quick test on SSPCloud
cd ~/work/Adversarial_examples
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')
"

# Test dataset access
python -c "
import os
print(f'Dataset exists: {os.path.exists(\"data/EuroSAT_RGB\")}')
if os.path.exists('data/EuroSAT_RGB'):
    classes = os.listdir('data/EuroSAT_RGB')
    print(f'Found {len(classes)} classes: {classes}')
"
```

## 📝 Notes for Your Project

1. **Use RGB version** (`EuroSAT_RGB`) for your initial experiments
2. **Implement models yourself** as required by your teacher
3. **Start with SimpleCNN** before moving to ResNet
4. **SSPCloud persistence**: Your project in `~/work/` will be saved between sessions

## 🔧 For Adversarial Attacks (Next Steps)

After getting basic models working, add:
1. FGSM attack implementation
2. PGD attack implementation  
3. Adversarial training
4. Robustness evaluation metrics

---