"""
datasets/feature_analysis.py
Module for generating datasets for Ilyas et al. experiments
"""
import torch
import torch.nn as nn
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from models import ResNet18

from attacks.pgd import PGD
from attacks.fgsm import FGSM
from config import DEVICE, STD, MEAN


class FeatureAnalysisDatasetGenerator:
    """Generate datasets for Ilyas et al. feature analysis experiments"""
    
    def __init__(self, base_model, device=DEVICE, seed=42):
        self.base_model = base_model.to(device)
        self.base_model.eval()
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)  # ← Set PyTorch seed
        random.seed(seed)  # ← Set Python random seed
    
    def create_non_robust_dataset(self,
                                 clean_train_path, 
                                 save_path,
                                 attack_type='fgsm',
                                 epsilon=0.01,
                                 alpha=0.002,
                                 iterations=5,
                                 mislabel_strategy='random',
                                 attack_strategy='target'):
        """
        Create a 'non-robust' dataset (experiment 2b from Ilyas et al.)
        
        Args:
            clean_train_path: Path to clean training images
            save_path: Where to save the non-robust dataset
            attack_type: 'fgsm' or 'pgd'
            epsilon: Attack strength
            alpha: Step size for PGD
            iterations: Number of PGD steps
            mislabel_strategy: How to mislabel ('random' or 'adversarial')
        """
        from data_loader.dataset import EuroSatDataset
        from torch.utils.data import DataLoader
        
        print("Creating non-robust dataset...")
        
        # Load clean dataset
        dataset = EuroSatDataset(root_dir=clean_train_path, train=True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Prepare attack
        if attack_type.lower() == 'fgsm':
            attack = FGSM(self.base_model, epsilon=epsilon, device=self.device)
        elif attack_type.lower() == 'pgd':
            epsilon_tensor = torch.tensor([epsilon / s for s in STD]).view(1,3,1,1).to(self.device)
            alpha_tensor = torch.tensor([alpha / s for s in STD]).view(1,3,1,1).to(self.device)
            attack = PGD(self.base_model, epsilon=epsilon_tensor, alpha=alpha_tensor, 
                        iterations=iterations, device=self.device)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Create save directory structure
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        config_dir = save_dir.parent / f"{save_dir.name}_config"
        config_dir.mkdir(exist_ok=True)

        # Initialize counters for mislabeling
        class_names = dataset.classes
        n_classes = len(class_names)
        
        # Create class directories
        for cls_name in class_names:
            (save_dir / cls_name).mkdir(exist_ok=True)
        
        img_idx = 0
        
        for batch_idx, (images, true_labels) in enumerate(tqdm(dataloader, desc="Generating non-robust dataset")):
            images = images.to(self.device)
            true_labels = true_labels.to(self.device)

            # with torch.enable_grad():
            #    adv_images = attack.attack(images, true_labels, target_labels=None)

            # Determine incorrect labels
            if mislabel_strategy == 'random':
                # Random incorrect labels (different from true label)
                batch_size = images.size(0)
                incorrect_labels = torch.randint(0, n_classes, (batch_size,), device=self.device)
                
                # Ensure incorrect labels are different from true labels
                mask = (incorrect_labels == true_labels)
                while mask.any():
                    incorrect_labels[mask] = torch.randint(0, n_classes, (mask.sum(),), device=self.device)
                    mask = (incorrect_labels == true_labels)
                
                target_labels = incorrect_labels if attack_strategy == 'target' else None

                # Generate adversarial perturbations
                with torch.enable_grad():
                    adv_images = attack.attack(images, true_labels, target_labels=target_labels)

            elif mislabel_strategy == 'adversarial':
                with torch.enable_grad():
                    adv_images = attack.attack(images, true_labels)
                # Use model's prediction on adversarial examples as incorrect labels
                with torch.no_grad():
                    outputs = self.base_model(adv_images)
                    incorrect_labels = torch.argmax(outputs, dim=1)
            
            # Save images with incorrect labels
            for i in range(adv_images.size(0)):
                incorrect_label = incorrect_labels[i].item()
                class_name = class_names[incorrect_label]
                
                save_path_img = save_dir / class_name / f"nonrobust_{img_idx}.png"
                save_image(adv_images[i], save_path_img)
                img_idx += 1
        
        print(f"Non-robust dataset saved to: {save_path}")
        print(f"Total images: {img_idx}")
        
        # Save dataset configuration
        config = {
            'dataset_type': 'non_robust',
            'source_dataset': clean_train_path,
            'attack_type': attack_type,
            'epsilon': epsilon,
            'alpha': alpha if attack_type == 'pgd' else None,
            'iterations': iterations if attack_type == 'pgd' else None,
            'mislabel_strategy': mislabel_strategy,
            'num_images': img_idx
        }
        
        config_path = config_dir / 'dataset_config.json'
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return save_dir
    
    def create_random_noise_dataset(self,
                                   clean_train_path,
                                   save_path,
                                   epsilon=0.01):
        """
        Create a random noise dataset (control experiment)
        
        Args:
            clean_train_path: Path to clean training images
            save_path: Where to save the random noise dataset
            epsilon: Noise magnitude (same as adversarial epsilon)
        """
        from data_loader.dataset import EuroSatDataset
        from torch.utils.data import DataLoader
        
        print("Creating random noise dataset (control)...")
        
        # Load clean dataset
        dataset = EuroSatDataset(root_dir=clean_train_path, train=True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Create save directory
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_dir = save_dir.parent / f"{save_dir.name}_config"
        config_dir.mkdir(exist_ok=True)
        
        # Get class names
        class_names = dataset.classes
        n_classes = len(class_names)
        
        # Create class directories
        for cls_name in class_names:
            (save_dir / cls_name).mkdir(exist_ok=True)
        
        img_idx = 0
        
        for batch_idx, (images, true_labels) in enumerate(tqdm(dataloader, desc="Generating random noise dataset")):
            images = images.to(self.device)
            batch_size = images.size(0)
            
            # Generate random noise
            noise = torch.randn_like(images) * epsilon
            noisy_images = torch.clamp(images + noise, 0, 1)
            
            # Assign random incorrect labels
            incorrect_labels = torch.randint(0, n_classes, (batch_size,), device=self.device)
            
            # Ensure labels are different from true labels
            mask = (incorrect_labels == true_labels.to(self.device))
            while mask.any():
                incorrect_labels[mask] = torch.randint(0, n_classes, (mask.sum(),), device=self.device)
                mask = (incorrect_labels == true_labels.to(self.device))
            
            # Save images with incorrect labels
            for i in range(noisy_images.size(0)):
                incorrect_label = incorrect_labels[i].item()
                class_name = class_names[incorrect_label]
                
                save_path_img = save_dir / class_name / f"random_{img_idx}.png"
                save_image(noisy_images[i], save_path_img)
                img_idx += 1
        
        print(f"Random noise dataset saved to: {save_path}")
        print(f"Total images: {img_idx}")
        
        # Save configuration
        config = {
            'dataset_type': 'random_noise',
            'source_dataset': clean_train_path,
            'epsilon': epsilon,
            'num_images': img_idx
        }
        
        config_path = config_dir / 'dataset_config.json'

        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return save_dir


def create_feature_analysis_summary(results_file='outputs/results/experiment_results.json'):
    """Create summary table and plots from collected results"""
    
    # Load collected results
    import json
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Convert to DataFrame
    rows = []
    for exp_name, config in results_data.items():
        row = {
            'Dataset': exp_name,
            'Train Acc': config.get('Train Acc', None),
            'Clean Test Acc': config.get('Clean Test Acc', None),
            'Adv Test Acc': config.get('Adv Test Acc', None),
        }
        
        # Calculate accuracy drop if possible
        if row['Clean Test Acc'] and row['Adv Test Acc']:
            row['Adv Acc Drop'] = row['Clean Test Acc'] - row['Adv Test Acc']
        else:
            row['Adv Acc Drop'] = None
        
        # Add dataset type for coloring
        if 'nonrobust' in exp_name:
            row['Type'] = 'Non-Robust'
            row['Color'] = '#FF6B6B'  # Red
        elif 'random' in exp_name:
            row['Type'] = 'Random Noise'
            row['Color'] = '#4ECDC4'  # Teal
        elif 'madry' in exp_name:
            row['Type'] = 'Adversarial Training'
            row['Color'] = '#45B7D1'  # Blue
        elif 'baseline' in exp_name:
            row['Type'] = 'Baseline'
            row['Color'] = '#96CEB4'  # Green
        else:
            row['Type'] = 'Other'
            row['Color'] = '#FFEAA7'  # Yellow
        
        rows.append(row)
    
    df = pd.DataFrame(rows)

    if df.empty:
        print("results_data seems empty")
    
    # Sort for better visualization
    order = ['Baseline', 'Adversarial Training', 'Non-Robust', 'Random Noise', 'Other']
    df['Type'] = pd.Categorical(df['Type'], categories=order, ordered=True)
    df = df.sort_values('Type')
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Clean vs Adversarial Accuracy (Grouped by Type)
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    
    clean_bars = ax.bar(x - width/2, df['Clean Test Acc'], width, 
                       label='Clean Accuracy', color='#3498db', alpha=0.8)
    adv_bars = ax.bar(x + width/2, df['Adv Test Acc'], width, 
                      label='Adversarial Accuracy', color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars
    for i, (clean_val, adv_val) in enumerate(zip(df['Clean Test Acc'], df['Adv Test Acc'])):
        if pd.notnull(clean_val):
            ax.text(i - width/2, clean_val + 1, f'{clean_val:.1f}', 
                   ha='center', va='bottom', fontsize=9)
        if pd.notnull(adv_val):
            ax.text(i + width/2, adv_val + 1, f'{adv_val:.1f}', 
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Clean vs Adversarial Test Accuracy\n(Key Result: Non-Robust > Random Noise)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Dataset'], rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Plot 2: Training Accuracy
    ax = axes[0, 1]
    colors = [row['Color'] for _, row in df.iterrows()]
    bars = ax.bar(df['Dataset'], df['Train Acc'], color=colors, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, df['Train Acc']):
        if pd.notnull(val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax.set_title('Final Training Accuracy\n(Non-Robust should train well)', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(df['Dataset'], rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = []
    for type_name, color in zip(df['Type'].unique(), df['Color'].unique()):
        legend_elements.append(Patch(facecolor=color, label=type_name, alpha=0.8))
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Plot 3: Adversarial Vulnerability (Accuracy Drop)
    ax = axes[1, 0]
    bars = ax.bar(df['Dataset'], df['Adv Acc Drop'], 
                  color=[row['Color'] for _, row in df.iterrows()], alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, df['Adv Acc Drop']):
        if pd.notnull(val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Adversarial Vulnerability\n(Lower is better)', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(df['Dataset'], rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, df['Adv Acc Drop'].max() * 1.2 if df['Adv Acc Drop'].max() > 0 else 100)
    
    # Plot 4: Key Comparison (Non-Robust vs Random Noise vs Baseline)
    ax = axes[1, 1]
    
    # Filter for key experiments
    key_experiments = ['baseline', 'nonrobust_fgsm', 'nonrobust_pgd', 'random_noise']
    key_df = df[df['Dataset'].isin(key_experiments)].copy()
    
    if len(key_df) >= 2:  # At least 2 experiments to compare
        x_key = np.arange(len(key_df))
        width_key = 0.25
        
        # Only use metrics that exist in the DataFrame
        available_metrics = []
        available_colors = []
        available_labels = []
        
        metric_configs = [
            ('Clean Test Acc', '#3498db', 'Clean Test'),
            ('Adv Test Acc', '#e74c3c', 'Adv Test'),
            ('Train Acc', '#2ecc71', 'Train')
        ]
        
        for metric, color, label in metric_configs:
            if metric in key_df.columns and key_df[metric].notna().any():
                available_metrics.append(metric)
                available_colors.append(color)
                available_labels.append(label)
        
        if available_metrics:  # Only plot if we have data
            for i, (metric, color, label) in enumerate(zip(available_metrics, available_colors, available_labels)):
                # Get values for this metric
                values = key_df[metric].values
                
                # Calculate position offset
                offset = (i - (len(available_metrics)-1)/2) * width_key
                
                bars = ax.bar(x_key + offset, values, width_key, 
                             color=color, alpha=0.8, label=label)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    if pd.notnull(val):
                        ax.text(bar.get_x() + bar.get_width()/2., val + 1,
                               f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Experiment', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title('Key Ilyas et al. Comparison\n(Non-Robust > Random Noise proves features exist)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x_key)
            ax.set_xticklabels(key_df['Dataset'], fontsize=10)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
        else:
            # No data available
            ax.text(0.5, 0.5, 'No metric data available\nfor selected experiments', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Key Comparison\n(No data)', fontsize=14, fontweight='bold')
            ax.axis('off')
    else:
        # Not enough experiments
        ax.text(0.5, 0.5, f'Insufficient experiments\nNeed at least 2, have {len(key_df)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Key Comparison\n(Need more experiments)', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path('outputs/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_dir / 'feature_analysis_summary.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nFeature analysis summary saved to: {fig_path}")
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM ILYAS ET AL. EXPERIMENTS")
    print("="*80)
    
    # Check if non-robust performs better than random noise
    if 'nonrobust_fgsm' in df['Dataset'].values and 'random_noise' in df['Dataset'].values:
        nonrobust_acc = df[df['Dataset'] == 'nonrobust_fgsm']['Clean Test Acc'].values[0]
        random_acc = df[df['Dataset'] == 'random_noise']['Clean Test Acc'].values[0]
        
        if pd.notnull(nonrobust_acc) and pd.notnull(random_acc):
            print(f"\n1. Non-Robust Dataset Accuracy: {nonrobust_acc:.2f}%")
            print(f"2. Random Noise Dataset Accuracy: {random_acc:.2f}%")
            
            if nonrobust_acc > random_acc + 10:  # Significant margin
                print(f"✓ CONCLUSIVE: Non-robust features exist and are predictive!")
                print(f"  → Adversarial perturbations contain learnable features")
                print(f"  → Accuracy difference: {nonrobust_acc - random_acc:.2f}% points")
            elif nonrobust_acc > random_acc:
                print(f"✓ INDICATIVE: Evidence of non-robust features")
                print(f"  → Adversarial perturbations may contain some features")
                print(f"  → Accuracy difference: {nonrobust_acc - random_acc:.2f}% points")
            else:
                print(f"✗ INCONCLUSIVE: No clear evidence of non-robust features")
                print(f"  → Try larger epsilon or different attack parameters")
    
    # Save detailed results
    csv_path = output_dir / 'feature_analysis_details.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    return df


def visualize_non_robust_features(experiment_name='nonrobust_fgsm', num_samples=16):
    """Visualize what the non-robust model has learned"""
    import torch
    import torch.nn.functional as F
    from models import ResNet18
    from data_loader.dataset import EuroSatDataset
    from torch.utils.data import DataLoader
    from config import DEVICE, MEAN, STD
    import numpy as np
    
    # Load the non-robust model
    model_path = f'outputs/models/{experiment_name}/best_model.pth'
    device = DEVICE
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    model = ResNet18().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load clean test dataset
    test_dataset = EuroSatDataset(
        root_dir='datasets/EuroSAT_RGB/test_clean',
        train=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=num_samples, 
        shuffle=True,
        num_workers=2
    )
    
    # Get a batch of clean test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Get model's predictions and confidence
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)
    
    # Calculate accuracy on this batch
    correct = (predictions == labels).sum().item()
    batch_accuracy = 100.0 * correct / len(labels)
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    class_names = test_dataset.classes
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array(STD) + np.array(MEAN)
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        true_class = class_names[labels[i].item()]
        pred_class = class_names[predictions[i].item()]
        confidence = confidences[i].item()
        
        # Create title with color coding
        if predictions[i] == labels[i]:
            title_color = 'green'
            marker = '✓'
        else:
            title_color = 'red'
            marker = '✗'
        
        title = f"True: {true_class}\nPred: {pred_class} {marker}\nConf: {confidence:.3f}"
        axes[i].set_title(title, color=title_color, fontsize=10)
        axes[i].axis('off')
        
        # Add colored border based on correctness
        axes[i].set_frame_on(True)
        axes[i].patch.set_edgecolor(title_color)
        axes[i].patch.set_linewidth(3)
    
    # Remove empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(
        f'Non-Robust Model ({experiment_name}) Predictions on Clean Test Images\n'
        f'Batch Accuracy: {batch_accuracy:.1f}% | Expected: > Random Chance (10%)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    # Save the visualization
    output_dir = Path('outputs/analysis/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f'{experiment_name}_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {save_path}")
    print(f"Batch accuracy: {batch_accuracy:.1f}%")
    
    # Statistical analysis
    print(f"\nStatistical Analysis:")
    print(f"- Random chance accuracy: {100.0/len(class_names):.1f}%")
    print(f"- This batch accuracy: {batch_accuracy:.1f}%")
    
    if batch_accuracy > 100.0/len(class_names) + 10:
        print("✓ Statistically significant evidence of learned features!")
    elif batch_accuracy > 100.0/len(class_names):
        print("✓ Some evidence of learned features")
    else:
        print("✗ No clear evidence of learned features above random chance")
    
    return batch_accuracy