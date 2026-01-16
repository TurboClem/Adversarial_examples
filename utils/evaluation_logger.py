import os
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader


class EvaluationLogger:
    """Unified system for logging evaluation results"""
    
    def __init__(self, base_dir: str = 'outputs/evaluation_logs'):
        """
        Args:
            base_dir: Base directory for all evaluation logs
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def evaluate_and_log(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        experiment_name: str,
        dataset_name: str,
        dataset_path: str,
        model_name: str = "resnet18",
        device: str = "cuda",
        save_plots_path: Optional[str] = None,
        additional_info: Optional[Dict] = None
    ) -> Tuple[float, str, str]:
        """
        Evaluate model and log comprehensive results
        
        Returns:
            (accuracy, log_file_path, json_file_path)
        """
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Run evaluation
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        accuracy = 100. * (sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels))
        
        # Log results
        log_file, json_file = self._log_evaluation_results(
            accuracy=accuracy,
            predictions=all_preds,
            true_labels=all_labels,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            model_name=model_name,
            class_names=test_loader.dataset.classes,
            save_plots_path=save_plots_path,
            additional_info=additional_info
        )
        
        return accuracy, log_file, json_file
    
    def _log_evaluation_results(
        self,
        accuracy: float,
        predictions: List[int],
        true_labels: List[int],
        experiment_name: str,
        dataset_name: str,
        dataset_path: str,
        model_name: str,
        class_names: List[str],
        save_plots_path: Optional[str] = None,
        additional_info: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Log evaluation results to files
        
        Returns:
            (log_file_path, json_file_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine where to save logs
        if save_plots_path:
            # Save in the experiment's plot directory
            log_dir = save_plots_path
        else:
            # Save in central evaluation logs directory
            log_dir = os.path.join(self.base_dir, experiment_name)
        
        os.makedirs(log_dir, exist_ok=True)
        
        # 1. Create human-readable log file
        log_file = os.path.join(log_dir, f'evaluation_{timestamp}.log')
        self._create_human_readable_log(
            log_file=log_file,
            accuracy=accuracy,
            predictions=predictions,
            true_labels=true_labels,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            model_name=model_name,
            class_names=class_names,
            timestamp=timestamp
        )
        
        # 2. Create JSON with all details
        json_file = os.path.join(log_dir, f'evaluation_{timestamp}.json')
        self._create_json_log(
            json_file=json_file,
            accuracy=accuracy,
            predictions=predictions,
            true_labels=true_labels,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            model_name=model_name,
            class_names=class_names,
            timestamp=timestamp,
            additional_info=additional_info
        )
        
        # 3. Create simple log for ResultCollector (backward compatibility)
        simple_log_file = os.path.join(log_dir, 'evaluation.log')
        with open(simple_log_file, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Test Accuracy: {accuracy:.2f}%\n")
            f.write(f"Total Samples: {len(true_labels)}\n")
        
        # 4. Update master CSV
        self._update_master_csv(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            model_name=model_name,
            accuracy=accuracy,
            timestamp=timestamp
        )
        
        print(f"✓ Evaluation logs saved to: {log_dir}")
        print(f"  - Human-readable: {log_file}")
        print(f"  - JSON details: {json_file}")
        print(f"  - Simple log: {simple_log_file}")
        
        return log_file, json_file
    
    def _create_human_readable_log(
        self,
        log_file: str,
        accuracy: float,
        predictions: List[int],
        true_labels: List[int],
        experiment_name: str,
        dataset_name: str,
        dataset_path: str,
        model_name: str,
        class_names: List[str],
        timestamp: str
    ):
        """Create human-readable evaluation log"""
        with open(log_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Path: {dataset_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Test Accuracy: {accuracy:.2f}%\n")
            f.write(f"Total Samples: {len(true_labels)}\n")
            f.write("\n")
            
            # Classification report
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 40 + "\n")
            
            report = classification_report(true_labels, predictions, 
                                         target_names=class_names, output_dict=False)
            f.write(report)
            f.write("\n\n")
            
            # Per-class accuracy
            f.write("PER-CLASS ACCURACY\n")
            f.write("-" * 40 + "\n")
            
            report_dict = classification_report(true_labels, predictions, 
                                              target_names=class_names, output_dict=True)
            
            for class_name in class_names:
                acc = report_dict[class_name]['recall'] * 100
                support = report_dict[class_name]['support']
                f.write(f"{class_name:25} {acc:6.2f}% ({support} samples)\n")
            
            f.write("\n")
            
            # Confusion matrix summary
            f.write("CONFUSION MATRIX SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            cm = confusion_matrix(true_labels, predictions)
            for i in range(len(class_names)):
                correct = cm[i, i]
                total = cm[i, :].sum()
                class_acc = 100.0 * correct / total if total > 0 else 0
                f.write(f"{class_names[i]:25} {class_acc:6.2f}% ({correct}/{total})\n")
            
            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("END OF EVALUATION\n")
            f.write("=" * 60 + "\n")
    
    def _create_json_log(
        self,
        json_file: str,
        accuracy: float,
        predictions: List[int],
        true_labels: List[int],
        experiment_name: str,
        dataset_name: str,
        dataset_path: str,
        model_name: str,
        class_names: List[str],
        timestamp: str,
        additional_info: Optional[Dict] = None
    ):
        """Create detailed JSON log"""
        report = classification_report(true_labels, predictions, 
                                     target_names=class_names, output_dict=True)
        cm = confusion_matrix(true_labels, predictions)
        
        results_dict = {
            'metadata': {
                'experiment_name': experiment_name,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'dataset_path': dataset_path,
                'timestamp': timestamp,
                'evaluation_type': 'clean' if 'clean' in dataset_name.lower() else 'adversarial',
            },
            'metrics': {
                'overall_accuracy': accuracy,
                'total_samples': len(true_labels),
                'per_class_accuracy': {},
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
            },
            'predictions': {
                'true_labels': true_labels,
                'predicted_labels': predictions,
            }
        }
        
        # Add per-class details
        for class_name in class_names:
            results_dict['metrics']['per_class_accuracy'][class_name] = {
                'accuracy': report[class_name]['recall'] * 100,
                'precision': report[class_name]['precision'] * 100,
                'f1_score': report[class_name]['f1-score'] * 100,
                'support': report[class_name]['support']
            }
        
        # Add additional info if provided
        if additional_info:
            results_dict['additional_info'] = additional_info
        
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=4, default=str)
    
    def _update_master_csv(
        self,
        experiment_name: str,
        dataset_name: str,
        model_name: str,
        accuracy: float,
        timestamp: str
    ):
        """Update master CSV with all evaluation results"""
        csv_file = os.path.join(self.base_dir, 'all_evaluations.csv')
        
        # Create new row
        new_row = {
            'timestamp': timestamp,
            'experiment': experiment_name,
            'dataset': dataset_name,
            'model': model_name,
            'accuracy': accuracy,
        }
        
        # Load existing or create new
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])
        
        # Save updated CSV
        df.to_csv(csv_file, index=False)
    
    def get_latest_accuracy(self, experiment_name: str, dataset_name: Optional[str] = None) -> Optional[float]:
        """Get latest accuracy for an experiment"""
        csv_file = os.path.join(self.base_dir, 'all_evaluations.csv')
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            if dataset_name:
                mask = (df['experiment'] == experiment_name) & (df['dataset'] == dataset_name)
            else:
                mask = df['experiment'] == experiment_name
            
            exp_results = df[mask]
            
            if not exp_results.empty:
                return exp_results.iloc[-1]['accuracy']
        
        return None


# Convenience function for backward compatibility
def log_evaluation_results(accuracy, predictions, true_labels, class_names,
                          experiment_name, dataset_path, model_name,
                          save_dir, dataset_name=None):
    """Backward compatible function"""
    logger = EvaluationLogger()
    
    if dataset_name is None:
        # Extract dataset name from path
        dataset_name = os.path.basename(dataset_path)
        if 'pgd' in dataset_name.lower() or 'adv' in dataset_name.lower():
            dataset_name = 'adversarial'
        else:
            dataset_name = 'clean'
    
    log_file, json_file = logger._log_evaluation_results(
        accuracy=accuracy,
        predictions=predictions,
        true_labels=true_labels,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        model_name=model_name,
        class_names=class_names,
        save_plots_path=save_dir
    )
    
    return log_file, json_file