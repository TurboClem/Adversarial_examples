"""
Visualization utilities for SSPCloud VM (headless environment)
All plots are automatically saved as PNG files instead of being displayed
"""
import matplotlib
# Use Agg backend for headless environments (no display)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import SAVE_PLOTS_PATH, MEAN, STD


def plot_training_history(history, model_name="Model"):
    """Plot training and validation metrics and save to file"""
    os.makedirs(SAVE_PLOTS_PATH, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - Accuracy Progress', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Combined (dual y-axis)
    ax2 = axes[2]
    ax2_twin = ax2.twinx()
    
    line1, = ax2.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    line2, = ax2.plot(history['val_loss'], 'b--', label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', color='b', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='b')
    
    line3, = ax2_twin.plot(history['train_acc'], 'r-', label='Train Acc', linewidth=2)
    line4, = ax2_twin.plot(history['val_acc'], 'r--', label='Val Acc', linewidth=2)
    ax2_twin.set_ylabel('Accuracy (%)', color='r', fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    # Combined legend
    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax2.set_title(f'{model_name} - Combined View', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(SAVE_PLOTS_PATH, f'{model_name}_training_history.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to: {filename}")
    
    # Also save as PDF for better quality
    pdf_filename = os.path.join(SAVE_PLOTS_PATH, f'{model_name}_training_history.pdf')
    plt.savefig(pdf_filename, bbox_inches='tight')
    
    plt.close(fig)  # Close the figure to free memory


def visualize_predictions(model, data_loader, class_names, device, num_samples=16, save=True):
    """Visualize model predictions on sample images and save to file"""
    model.eval()
    
    # Get a batch of data
    images, labels = next(iter(data_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities.max(1)[0].cpu().numpy()
    
    # Denormalize images for display
    images_denorm = denormalize(images)
    
    # Calculate grid size
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    
    # Flatten axes array if needed
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(num_samples):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = axes[row, col]
        
        # Display image
        img = images_denorm[idx].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        
        # Set title with prediction results
        true_class = class_names[labels[idx].item()]
        pred_class = class_names[predicted[idx].item()]
        conf = confidence[idx]
        
        color = 'green' if labels[idx] == predicted[idx] else 'red'
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {conf:.2f}"
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Model Predictions on Sample Images', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        filename = os.path.join(SAVE_PLOTS_PATH, 'sample_predictions.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Predictions visualization saved to: {filename}")
        
        # Also save as PDF
        pdf_filename = os.path.join(SAVE_PLOTS_PATH, 'sample_predictions.pdf')
        plt.savefig(pdf_filename, bbox_inches='tight')
    
    plt.close(fig)
    
    # Calculate and print accuracy on this batch
    correct = (predicted == labels).sum().item()
    batch_accuracy = 100. * correct / num_samples
    print(f"Batch accuracy on {num_samples} samples: {batch_accuracy:.2f}%")
    
    return batch_accuracy


def denormalize(tensor):
    """Denormalize tensor images for visualization"""
    tensor = tensor.clone()
    
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    
    return torch.clamp(tensor, 0, 1)


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True):
    """Plot confusion matrix and save to file"""
    os.makedirs(SAVE_PLOTS_PATH, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix (counts)'
    
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2)
    
    # Create heatmap
    ax = sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Normalized Accuracy' if normalize else 'Count'},
                    square=True)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save the figure
    norm_suffix = '_normalized' if normalize else ''
    filename = os.path.join(SAVE_PLOTS_PATH, f'confusion_matrix{norm_suffix}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {filename}")
    
    # Also save as PDF
    pdf_filename = os.path.join(SAVE_PLOTS_PATH, f'confusion_matrix{norm_suffix}.pdf')
    plt.savefig(pdf_filename, bbox_inches='tight')
    
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return cm


def plot_sample_images(data_loader, class_names, num_classes=10, samples_per_class=3):
    """Plot sample images from each class in the dataset"""
    os.makedirs(SAVE_PLOTS_PATH, exist_ok=True)
    
    # Get a batch of data
    images, labels = next(iter(data_loader))
    
    # Create figure
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                             figsize=(3 * samples_per_class, 3 * num_classes))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalize all images once
    images_denorm = denormalize(images)
    
    # Track which classes we've plotted
    class_counts = {i: 0 for i in range(num_classes)}
    
    # Plot samples
    for idx in range(len(images)):
        if all(count >= samples_per_class for count in class_counts.values()):
            break
            
        label = labels[idx].item()
        if class_counts[label] < samples_per_class:
            row = label
            col = class_counts[label]
            
            ax = axes[row, col]
            img = images_denorm[idx].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.set_title(f"Class: {class_names[label]}\nSample {col+1}")
            ax.axis('off')
            
            class_counts[label] += 1
    
    plt.suptitle(f'EuroSat Dataset Samples ({num_classes} classes)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(SAVE_PLOTS_PATH, 'dataset_samples.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Dataset samples plot saved to: {filename}")
    
    plt.close(fig)


def plot_model_architecture(model, input_size=(3, 64, 64)):
    """Create a simple visualization of the model architecture"""
    os.makedirs(SAVE_PLOTS_PATH, exist_ok=True)
    
    try:
        from torchviz import make_dot
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size)
        
        # Forward pass
        output = model(dummy_input)
        
        # Create visualization
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        # Save as PNG and PDF
        png_filename = os.path.join(SAVE_PLOTS_PATH, 'model_architecture.png')
        dot.render(png_filename.replace('.png', ''), format='png', cleanup=True)
        
        pdf_filename = os.path.join(SAVE_PLOTS_PATH, 'model_architecture.pdf')
        dot.render(pdf_filename.replace('.pdf', ''), format='pdf', cleanup=True)
        
        print(f"Model architecture visualization saved to: {png_filename}")
        
    except ImportError:
        print("torchviz not installed. Install with: pip install torchviz")
        print("Model summary:")
        print(model)
        
        # Save text summary
        summary_filename = os.path.join(SAVE_PLOTS_PATH, 'model_summary.txt')
        with open(summary_filename, 'w') as f:
            f.write(str(model))
            f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
            f.write(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        print(f"Model summary saved to: {summary_filename}")


def save_training_metrics_to_csv(history, model_name="Model"):
    """Save training metrics to CSV file for further analysis"""
    os.makedirs(SAVE_PLOTS_PATH, exist_ok=True)
    
    import pandas as pd
    
    # Create DataFrame
    df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc']
    })
    
    # Save to CSV
    csv_filename = os.path.join(SAVE_PLOTS_PATH, f'{model_name}_training_metrics.csv')
    df.to_csv(csv_filename, index=False)
    
    print(f"Training metrics saved to CSV: {csv_filename}")
    return df


def create_training_report(history, model_name, test_accuracy, class_names, y_true=None, y_pred=None):
    """Create a comprehensive training report with multiple visualizations"""
    os.makedirs(SAVE_PLOTS_PATH, exist_ok=True)
    
    # Plot training history
    plot_training_history(history, model_name)
    
    # Save metrics to CSV
    save_training_metrics_to_csv(history, model_name)
    
    # Plot confusion matrix if data provided
    if y_true is not None and y_pred is not None:
        plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
        plot_confusion_matrix(y_true, y_pred, class_names, normalize=False)
    
    # Create report text file
    report_filename = os.path.join(SAVE_PLOTS_PATH, f'{model_name}_training_report.txt')
    with open(report_filename, 'w') as f:
        f.write(f"=== {model_name} Training Report ===\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n\n")
        
        f.write("Training History:\n")
        f.write("Epoch | Train Loss | Val Loss | Train Acc | Val Acc\n")
        f.write("-" * 60 + "\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"{epoch+1:5d} | {history['train_loss'][epoch]:10.4f} | "
                   f"{history['val_loss'][epoch]:8.4f} | {history['train_acc'][epoch]:9.2f} | "
                   f"{history['val_acc'][epoch]:7.2f}\n")
    
    print(f"Comprehensive training report saved to: {report_filename}")