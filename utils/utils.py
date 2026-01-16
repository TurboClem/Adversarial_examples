"""
Utility functions for AdvProp
"""
import torch
import numpy as np
import os


def create_advprop_report(history, model_name, epsilon, save_path):
    """Create a report for AdvProp training"""
    os.makedirs(save_path, exist_ok=True)
    
    report_file = os.path.join(save_path, f"{model_name}_advprop_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ADVPROP TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Alpha: {epsilon/4:.4f}\n")
        f.write(f"Training Epochs: {len(history['train_loss'])}\n\n")
        
        f.write("FINAL METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Clean Training Accuracy: {history['train_acc_clean'][-1]:.2f}%\n")
        f.write(f"Final Adversarial Training Accuracy: {history['train_acc_adv'][-1]:.2f}%\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%\n\n")
        
        f.write("BEST METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%\n")
        f.write(f"Best Epoch: {np.argmax(history['val_acc']) + 1}\n\n")
        
        f.write("ACCURACY GAP ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        initial_gap = history['train_acc_clean'][0] - history['train_acc_adv'][0]
        final_gap = history['train_acc_clean'][-1] - history['train_acc_adv'][-1]
        f.write(f"Initial Clean-Adv Gap: {initial_gap:.2f}%\n")
        f.write(f"Final Clean-Adv Gap: {final_gap:.2f}%\n")
        f.write(f"Gap Reduction: {initial_gap - final_gap:.2f}%\n")
    
    print(f"AdvProp report saved to: {report_file}")
    return report_file