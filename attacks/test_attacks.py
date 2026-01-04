"""
Test script for adversarial attacks on EuroSat classification
"""
import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from models.resnet import ResNet18
from models.simple_cnn import SimpleCNN
from data_loader.dataset import EuroSatDataset
from attacks import FGSM, PGD, visualize_attacks, create_attack_summary
from config import IMG_SIZE, MEAN, STD, BATCH_SIZE


def load_model_and_data(model_type='resnet18'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if model_type == 'resnet18':
        model = ResNet18().to(device)
        model_path = 'outputs/models/resnet18_final.pth'
    elif model_type == 'simple_cnn':
        model = SimpleCNN().to(device)
        model_path = 'outputs/models/simple_cnn_final.pth'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f" Model loaded from {model_path}")
    else:
        print(f"Model not found at {model_path}, using random weights")
    
    model.eval()
    
    # Test dataset
    dataset = EuroSatDataset(
        root_dir='./data/EuroSAT_RGB',
        transform=None,
        train=False
    )
    
    test_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    return model, test_loader, device, dataset.classes


def demo_single_attack():
    """Demo FGSM and PGD on a few images"""
    model, test_loader, device, class_names = load_model_and_data('resnet18')
    
    # Get one batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Take only 4 images for visualization
    images = images[:4]
    labels = labels[:4]
    
    print(f"\n{'='*60}")
    print("DEMO: SINGLE BATCH ATTACK")
    print(f"{'='*60}")
    
    # 1. Test FGSM
    print("\n1. FGSM Attack (ε=0.03):")
    fgsm = FGSM(model, epsilon=0.03)
    adv_fgsm = fgsm.attack(images, labels)
    
    with torch.no_grad():
        preds_clean = torch.argmax(model(images), dim=1)
        preds_fgsm = torch.argmax(model(adv_fgsm), dim=1)
    
    # 2. Test PGD
    print("2. PGD Attack (ε=0.03, α=0.01, iter=10):")
    pgd = PGD(model, epsilon=0.03, alpha=0.01, iterations=10)
    adv_pgd = pgd.attack(images, labels)
    
    with torch.no_grad():
        preds_pgd = torch.argmax(model(adv_pgd), dim=1)
    
    # Display results
    print("\nResults on 4 images:")
    print("-" * 40)
    for i in range(len(images)):
        print(f"Image {i+1}:")
        print(f"  True: {class_names[labels[i]]}")
        print(f"  Clean pred: {class_names[preds_clean[i]]} {'✓' if preds_clean[i] == labels[i] else '✗'}")
        print(f"  FGSM pred: {class_names[preds_fgsm[i]]} {'✓' if preds_fgsm[i] == labels[i] else '✗'}")
        print(f"  PGD pred: {class_names[preds_pgd[i]]} {'✓' if preds_pgd[i] == labels[i] else '✗'}")
        print()
    
 
    visualize_attacks(
        images, adv_fgsm,
        preds_clean, preds_fgsm,
        labels, class_names,
        filename='outputs/plots/fgsm_attack_demo.png'
    )
    
    visualize_attacks(
        images, adv_pgd,
        preds_clean, preds_pgd,
        labels, class_names,
        filename='outputs/plots/pgd_attack_demo.png'
    )


def evaluate_all_attacks():
    """Evaluate model against multiple attack configurations"""
    model, test_loader, device, class_names = load_model_and_data('resnet18')
    
    print(f"\n{'='*60}")
    print("EVALUATION: MULTIPLE ATTACK CONFIGURATIONS")
    print(f"{'='*60}")
    
    # Define different attacks to test
    attacks_dict = {
        'FGSM (ε=0.01)': FGSM(model, epsilon=0.01),
        'FGSM (ε=0.03)': FGSM(model, epsilon=0.03),
        'FGSM (ε=0.05)': FGSM(model, epsilon=0.05),
        'PGD (ε=0.03, iter=10)': PGD(model, epsilon=0.03, alpha=0.01, iterations=10),
        'PGD (ε=0.03, iter=20)': PGD(model, epsilon=0.03, alpha=0.005, iterations=20),
        'PGD (ε=0.05, iter=10)': PGD(model, epsilon=0.05, alpha=0.02, iterations=10),
    }
    
   
    df = create_attack_summary(model, test_loader, attacks_dict, device)
    
    # Save results
    results_dir = 'outputs/results'
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(f'{results_dir}/robustness_evaluation.csv', index=False)
    print(f"\nResults saved to {results_dir}/robustness_evaluation.csv")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    attacks = df['Attack'].tolist()
    clean_acc = [float(x.strip('%')) for x in df['Clean Accuracy']]
    adv_acc = [float(x.strip('%')) for x in df['Adversarial Accuracy']]
    
    x = range(len(attacks))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], clean_acc, width, label='Clean Accuracy', color='skyblue')
    plt.bar([i + width/2 for i in x], adv_acc, width, label='Adversarial Accuracy', color='lightcoral')
    
    plt.xlabel('Attack Configuration')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Robustness Against Different Attacks')
    plt.xticks(x, attacks, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/robustness_comparison.png', dpi=150)
    plt.close()
    
    print(f"Plot saved to {results_dir}/robustness_comparison.png")


def compare_models():
    """Compare robustness of SimpleCNN vs ResNet"""
    print(f"\n{'='*60}")
    print("COMPARISON: SimpleCNN vs ResNet18")
    print(f"{'='*60}")
    
    results = []
    
    for model_name in ['simple_cnn', 'resnet18']:
        print(f"\nEvaluating {model_name}...")
        model, test_loader, device, _ = load_model_and_data(model_name)
        
        # Test with strong PGD attack
        attack = PGD(model, epsilon=0.03, alpha=0.01, iterations=10)
        
        clean_acc, adv_acc, success_rate = evaluate_robustness(
            model, test_loader[:20], attack, device  # Use subset for speed
        )
        
        results.append({
            'Model': model_name,
            'Clean Accuracy': f"{clean_acc*100:.2f}%",
            'Adversarial Accuracy': f"{adv_acc*100:.2f}%",
            'Success Rate': f"{success_rate*100:.2f}%"
        })
    
    # Display comparison
    print("\nModel Comparison:")
    print("-" * 60)
    for res in results:
        print(f"{res['Model']:12} | Clean: {res['Clean Accuracy']:10} | Adv: {res['Adversarial Accuracy']:10} | Success: {res['Success Rate']}")


if __name__ == "__main__":
    # Create output directories
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    
    print(" Adversarial Attacks Testing")
    print("=" * 60)
    
    # Run demos
    demo_single_attack()
    evaluate_all_attacks()
    compare_models()
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"Check outputs/plots/ for visualizations")
    print(f"Check outputs/results/ for detailed metrics")