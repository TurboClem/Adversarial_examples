"""
Main script for EuroSat adversarial robustness project
"""
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from models import create_simple_cnn, ResNet18
# AJOUTER ces imports
from models.resnet_with_advprop import ResNet18AdvProp
from train.advprop_trainer import AdvPropTrainer

from train.trainer import ModelTrainer, create_data_loaders
from utils.visualization import plot_training_history, visualize_predictions, create_training_report, plot_sample_images


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EuroSat Adversarial Robustness Project')
    
    # MODIFIER les choix du modèle
    parser.add_argument('--model', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'resnet18', 'resnet18_advprop'],  # AJOUTER
                       help='Model architecture to use')
    
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--data-path', type=str, default=DATA_PATH,
                       help='Path to EuroSat dataset')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the model')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    
    # NOUVEAUX ARGUMENTS pour AdvProp
    parser.add_argument('--advprop', action='store_true',
                       help='Use AdvProp training method')
    parser.add_argument('--epsilon', type=float, default=0.03,
                       help='Perturbation strength for adversarial training')
    parser.add_argument('--pgd-iter', type=int, default=7,
                       help='PGD iterations for AdvProp')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directories
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    os.makedirs(SAVE_PLOTS_PATH, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    print(f"Data path: {args.data_path}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size
    )
    
    # Create model - MODIFIER cette section
    print(f"\nCreating {args.model} model...")
    if args.model == 'simple_cnn':
        model = create_simple_cnn(num_classes=len(class_names))
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=len(class_names))
    elif args.model == 'resnet18_advprop':
        model = ResNet18AdvProp(num_classes=len(class_names))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer - MODIFIER selon le modèle/méthode
    if args.advprop or args.model == 'resnet18_advprop':
        print(f"Using AdvProp trainer (ε={args.epsilon}, iter={args.pgd_iter})")
        trainer = AdvPropTrainer(
            model=model,
            epsilon=args.epsilon,
            alpha=args.epsilon/3,  # Standard ratio
            iterations=args.pgd_iter,
            device=DEVICE
        )
    else:
        print("Using standard trainer")
        trainer = ModelTrainer(model, device=DEVICE)
    
    # Train model
    if args.train:
        print(f"\nTraining {args.model} for {args.epochs} epochs...")
        
        # Utiliser la méthode train appropriée
        history = trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr
        )
        
        # Plot training history
        model_name = args.model
        if args.advprop or args.model == 'resnet18_advprop':
            model_name += "_AdvProp"
        plot_training_history(history, model_name=model_name)
        
        # Save final model
        trainer.save_model(f'{model_name}_final.pth')

        # Evaluate on test set
        test_accuracy, predictions, true_labels = trainer.evaluate(test_loader)
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Create comprehensive report
        create_training_report(
            history=history,
            model_name=model_name,
            test_accuracy=test_accuracy,
            class_names=class_names,
            y_true=true_labels,
            y_pred=predictions
        )
    
    # Evaluate model
    if args.evaluate:
        print("\nEvaluating model on test set...")
        
        # Load best model if exists
        model_name = args.model
        if args.advprop or args.model == 'resnet18_advprop':
            model_name += "_AdvProp"
            
        model_path = os.path.join(SAVE_MODEL_PATH, f'best_model{"" if args.model == "resnet18_advprop" else ""}.pth')
        if os.path.exists(model_path):
            trainer.load_model(os.path.basename(model_path))
            print(f"Loaded best model from {model_path}")
        else:
            # Essayer le modèle final
            model_path = os.path.join(SAVE_MODEL_PATH, f'{model_name}_final.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=DEVICE)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded final model from {model_path}")
        
        accuracy, predictions, true_labels = trainer.evaluate(test_loader)
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Visualize predictions
    if args.visualize:
        print("\nVisualizing predictions...")
        
        # Load model if needed
        model_name = args.model
        if args.advprop or args.model == 'resnet18_advprop':
            model_name += "_AdvProp"
            
        model_path = os.path.join(SAVE_MODEL_PATH, f'{model_name}_final.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=DEVICE)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        visualize_predictions(
            trainer.model, test_loader, class_names, DEVICE
        )

        # Show dataset samples
        plot_sample_images(train_loader, class_names)
    
    print("\nDone!")


if __name__ == '__main__':
    main()