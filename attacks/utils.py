"""
Utility functions for adversarial attacks
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def denormalize(tensor, mean=None, std=None):
    """
    Denormalize a tensor normalized with ImageNet stats
    
    Args:
        tensor: normalized tensor [C, H, W] or [B, C, H, W]
        mean: normalization mean (default: ImageNet)
        std: normalization std (default: ImageNet)
    
    Returns:
        Denormalized tensor in range [0, 1]
    """
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406])
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225])
    
    # Reshape for broadcasting
    if tensor.dim() == 4:  # Batch
        mean = mean.view(1, 3, 1, 1).to(tensor.device)
        std = std.view(1, 3, 1, 1).to(tensor.device)
    else:  # Single image
        mean = mean.view(3, 1, 1).to(tensor.device)
        std = std.view(3, 1, 1).to(tensor.device)
    
    return tensor * std + mean


def visualize_attacks(clean_images, adv_images, 
                     clean_preds, adv_preds,
                     labels, class_names,
                     filename='attack_visualization.png',
                     max_images=5):
    """
    Visualize clean vs adversarial images
    
    Args:
        clean_images: original images
        adv_images: adversarial images
        clean_preds: predictions on clean images
        adv_preds: predictions on adversarial images
        labels: true labels
        class_names: list of class names
        filename: output filename
        max_images: maximum number of images to display
    """
    n_images = min(max_images, len(clean_images))
    
    fig, axes = plt.subplots(4, n_images, figsize=(3*n_images, 12))
    
    if n_images == 1:
        axes = axes.reshape(4, 1)
    
    for i in range(n_images):
        # Denormalize images
        clean_denorm = denormalize(clean_images[i].cpu()).permute(1, 2, 0).numpy()
        adv_denorm = denormalize(adv_images[i].cpu()).permute(1, 2, 0).numpy()
        
        # Perturbation
        perturbation = (adv_images[i] - clean_images[i]).cpu()
        perturbation = perturbation.permute(1, 2, 0).numpy()
        
        # Normalize perturbation for visualization
        perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
        
        # Row 1: Clean image
        axes[0, i].imshow(np.clip(clean_denorm, 0, 1))
        axes[0, i].set_title(f"Clean\nTrue: {class_names[labels[i]]}\nPred: {class_names[clean_preds[i]]}")
        axes[0, i].axis('off')
        
        # Row 2: Adversarial image
        axes[1, i].imshow(np.clip(adv_denorm, 0, 1))
        axes[1, i].set_title(f"Adversarial\nPred: {class_names[adv_preds[i]]}")
        axes[1, i].axis('off')
        
        # Row 3: Perturbation (amplified)
        axes[2, i].imshow(perturbation, cmap='seismic', vmin=0, vmax=1)
        axes[2, i].set_title("Perturbation")
        axes[2, i].axis('off')
        
        # Row 4: Difference (pixel-wise)
        diff = np.abs(clean_denorm - adv_denorm).mean(axis=2)
        axes[3, i].imshow(diff, cmap='hot')
        axes[3, i].set_title(f"L1 diff: {diff.mean():.4f}")
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {filename}")


def evaluate_robustness(model, test_loader, attack, device='cuda'):
    """
    Evaluate model robustness against an attack
    
    Args:
        model: model to evaluate
        test_loader: DataLoader for test set
        attack: attack instance
        device: device to run on
    
    Returns:
        clean_accuracy, adversarial_accuracy, success_rate
    """
    model.eval()
    total_correct_clean = 0
    total_correct_adv = 0
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = attack.attack(images, labels)
        
        # Predictions
        with torch.no_grad():
            # Clean predictions
            outputs_clean = model(images)
            preds_clean = torch.argmax(outputs_clean, dim=1)
            
            # Adversarial predictions
            outputs_adv = model(adv_images)
            preds_adv = torch.argmax(outputs_adv, dim=1)
        
        # Update counts
        total_correct_clean += (preds_clean == labels).sum().item()
        total_correct_adv += (preds_adv == labels).sum().item()
        total_samples += labels.size(0)
        
        # Progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(test_loader)}")
    
    clean_acc = total_correct_clean / total_samples
    adv_acc = total_correct_adv / total_samples
    success_rate = 1.0 - adv_acc
    
    return clean_acc, adv_acc, success_rate


def create_attack_summary(model, test_loader, attacks_dict, device='cuda'):
    """
    Create summary of model robustness against multiple attacks
    
    Args:
        model: model to evaluate
        test_loader: DataLoader for test set
        attacks_dict: dict of {attack_name: attack_instance}
        device: device to run on
    
    Returns:
        DataFrame with results
    """
    import pandas as pd
    
    results = []
    
    for attack_name, attack in attacks_dict.items():
        print(f"\nEvaluating {attack_name}...")
        clean_acc, adv_acc, success_rate = evaluate_robustness(
            model, test_loader, attack, device
        )
        
        results.append({
            'Attack': attack_name,
            'Clean Accuracy': f"{clean_acc*100:.2f}%",
            'Adversarial Accuracy': f"{adv_acc*100:.2f}%",
            'Success Rate': f"{success_rate*100:.2f}%",
            'Accuracy Drop': f"{(clean_acc - adv_acc)*100:.2f}%",
            'Epsilon': getattr(attack, 'epsilon', 'N/A'),
            'Iterations': getattr(attack, 'iterations', 1)
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    return df