"""
Main script for EuroSat adversarial robustness project
"""
import argparse
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from models import create_simple_cnn, ResNet18
from train.trainer import ModelTrainer, create_data_loaders
from utils.visualization import plot_training_history, visualize_predictions, plot_training_history, create_training_report, plot_sample_images


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EuroSat Adversarial Robustness Project')
    parser.add_argument('--model', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'resnet18'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                       help='Patience for early stopping during training')
    parser.add_argument('--madry', type=str, default=MADRY,
                       help='Madry adversarial training config as JSON')
    parser.add_argument('--seed', type=int, default=SEED,
                       help='Seed')
    parser.add_argument('--data-path-train', type=str, default=DATA_PATH_TRAIN,
                       help='Path to EuroSat Train dataset')
    parser.add_argument('--data-path-eval', type=str, default=DATA_PATH_EVAL,
                       help='Path to EuroSat Test dataset')
    parser.add_argument('--save-model-path', type=str, default=SAVE_MODEL_PATH,
                       help='Path where to save models')
    parser.add_argument('--save-plots-path', type=str, default=SAVE_PLOTS_PATH,
                       help='Path where to save plots')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the model')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.save_model_path, exist_ok=True)
    os.makedirs(args.save_plots_path, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    print(f"Data paths: train = {args.data_path_train}, eval = {args.data_path_eval}")
    
    # # Create data loaders
    print("\nLoading dataset...")
    if args.train or args.visualize:
        train_loader, val_loader, class_names = create_data_loaders(
            data_path=args.data_path_train,
            batch_size=args.batch_size,
            mode="train"
        )
    if args.evaluate or args.visualize:
        test_loader, class_names = create_data_loaders(
            data_path=args.data_path_eval,
            batch_size=args.batch_size,
            mode="eval"
        )

    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == 'simple_cnn':
        model = create_simple_cnn(num_classes=len(class_names))
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=len(class_names))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = ModelTrainer(model, device=DEVICE)

    # Train model
    if args.train:

        madry = None
        if args.madry is not None:
            madry = json.loads(args.madry)

        print(f"\nTraining {args.model} for {args.epochs} epochs...")
        history = trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            madry=madry,
            save_model_path=args.save_model_path
        )
        
        # Plot training history
        plot_training_history(history, model_name=args.model, save_plots_path=args.save_plots_path)
        
        # Save final model
        trainer.save_model(filename=f'{args.model}_final.pth', save_model_path=args.save_model_path)
        
        # Create comprehensive report
        create_training_report(
            history=history,
            model_name=args.model,
            class_names=class_names,
            save_plots_path=args.save_plots_path
        )
    
    # Evaluate model
    if args.evaluate:
        print("\nEvaluating model on test set...")
        
        # Load best model if exists
        model_path = os.path.join(args.save_model_path, 'best_model.pth')
        if os.path.exists(model_path):
            trainer.load_model(filename='best_model.pth', save_model_path=args.save_model_path)
            print("Loaded best model for evaluation")
        
        accuracy, predictions, true_labels = trainer.evaluate(test_loader)
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # You can add confusion matrix here if needed
        from utils.visualization import plot_confusion_matrix
        plot_confusion_matrix(true_labels, predictions, class_names, save_plots_path=args.save_plots_path)
    
    # Visualize predictions
    if args.visualize:
        print("\nVisualizing predictions...")
        
        # Load best model if exists
        model_path = os.path.join(args.save_model_path, 'best_model.pth')
        if os.path.exists(model_path):
            trainer.load_model(filename='best_model.pth', save_model_path=args.save_model_path)
        
        visualize_predictions(
            trainer.model, test_loader, class_names, DEVICE, 16, True, args.save_plots_path
        )

        # Show dataset samples
        plot_sample_images(train_loader, class_names)
    
    print("\nDone!")


if __name__ == '__main__':
    main()