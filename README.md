# EuroSat Adversarial Robustness Project

## 📋 Project Overview
This project studies adversarial robustness of image classification models on satellite imagery (EuroSat dataset). The goal is to implement and compare different CNN architectures, then analyze their vulnerability to adversarial attacks.
## 🏗️ Project Structure
```
Adversarial_examples/
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── pipfreeze.txt                 # Full pip freeze output for reproducibility
├── config.py                     # Configuration settings (hyperparameters, paths, etc.)
├── main.py                       # Main entry point for experiments
├── download_data.sh              # Script to download EuroSat dataset
├── 0_create_datasets_and_baseline.ipynb    # Creates Train/Test sets and Baseline model
├── 1_Ilyas.py                    # Runs Ilyas et al. experiment
├── 2_Madry.ipynb                 # Runs Madry et al. experiment (PGD adversarial training)
├── 3_Xie.ipynb                   # Runs Xie et al. experiment
├── README.md                     # Project documentation
│
├── attacks/                      # Adversarial attack implementations
│   ├── __init__.py
│   ├── base_attack.py            # Base class for all attacks
│   ├── fgsm.py                   # FGSM attack implementation
│   ├── pgd.py                    # PGD attack implementation (Madry et al.)
│   ├── test_attacks.py           # Test scripts for attacks
│   ├── utils.py                  # Attack-specific utilities
│   └── configs/                  # Attack configuration files
│       └── attack_config.json    # JSON configs for different attacks
│
├── data_loader/                  # Data handling utilities
│   ├── __init__.py
│   └── dataset.py                # EuroSat dataset class and data loaders
│
├── datasets/                     # Dataset analysis and preprocessing
│   ├── EuroSAT_RGB/              # EuroSat split in train/set (result of 0_create_datasets_and_baseline.ipynb)
│   └── feature_analysis.py       # Feature analysis of datasets
│
├── models/                       # Neural network model implementations
│   ├── __init__.py
│   ├── simple_cnn.py             # Custom CNN implementation (baseline)
│   ├── resnet.py                 # Standard ResNet implementation
│   └── resnet_with_advprop.py    # ResNet with AdvProp adversarial training
│
├── train/                        # Training frameworks
│   ├── __init__.py
│   ├── trainer.py                # Standard model trainer
│   └── advprop_trainer.py        # Trainer for AdvProp adversarial training
│
├── utils/                        # Helper functions and utilities
│   ├── utils.py                  # General utility functions
│   ├── visualization.py          # Plotting and visualization utilities
│   ├── evaluation_logger.py      # Logger for evaluation metrics
│   └── result_collector.py       # Collector for experimental results
│
├── outputs/                      # Generated outputs and results
│   ├── models/                   # Saved model checkpoints
│   ├── plots/                    # Training/validation plots
│   ├── evaluation_logs/          # Saved evaluation results
│   ├── results/                  # Experimental results
│   └── analysis/                 # Analysis outputs
│
└── data/EuroSAT_RGB/             # EuroSat dataset directory (when downloaded)
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    └── SeaLake/
```


## References

1. **Adversarial Examples Are Not Bugs, They Are Features**  
   Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, and Aleksander Madry.  
   In *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

2. **Towards Deep Learning Models Resistant to Adversarial Attacks**  
   Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.  
   In *International Conference on Learning Representations (ICLR)*, 2018.

3. **Adversarial Examples Improve Image Recognition**  
   Cihang Xie, Mingxing Tan, Boqing Gong, Jiang Wang, Alan Yuille, and Quoc V. Le.  
   In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

4. **EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification**  
   Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth.  
   In *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2019.

5. **Explaining and harnessing adversarial examples**  
   Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy.  
   In *International Conference on Learning Representations (ICLR)*, 2015.


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

- **Run the notebook ```0_create_datasets_and_baseline.ipynb```** to create the reproductible clean train and test sets, train the baseline model, and create the attacked test set.

### 3. Run the Project

To follow our framework:
- Run the notebook ```1_Ilyas.py``` to reproduce the experiment on Ilyas et al.
- Run the notebook ```2_Madry.ipynb``` to reproduce the experiment on Madry et al.
- Run the notebook ```3_Xie.py``` to reproduce the experiment on Xie et al.

#### 3.1 Ilyas et al. experiment.
The file ```1_Ilyas.py``` will :
- load the baseline.
- generate adversarial sets (using methods : FGSM method, PGD, random noise)
- train a model on each of adversarial train set.
- evaluate those models on adversarial test set and clean test set.
- produce metrics and plots.

#### 3.2 Madry et al. experiment.
The file ```2_Madry.ipynb``` will :
- train a model with the min-max optimization approach and save the model
- test on the clean test set and the attacked test set
- produce metrics and plots

#### 3.3 Xie et al. experiment.
The file ```3_Xie.ipynb``` will :
- train a model with the approach of Xie et al.
- test on the clean test set and the attacked test set
- produce metrics and plots

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
