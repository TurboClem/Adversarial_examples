import json
import pandas as pd
import os
from pathlib import Path
import torch
import re

class ResultCollector:
    """Collect results from training logs and model checkpoints"""
    
    def __init__(self, results_file='outputs/results/experiment_results.json'):
        self.results_file = results_file
        self.results = {}
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        # Load existing results
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.results = json.load(f)
    
    def extract_from_training_log(self, log_file):
        """Extract metrics from training log file"""
        metrics = {}
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract final training accuracy
            train_acc_match = re.search(r'Train Loss: [\d\.]+\s*\|\s*Train Acc: ([\d\.]+)%', content)
            if train_acc_match:
                metrics['Train Acc'] = float(train_acc_match.group(1))
            
            # Extract final validation accuracy
            val_acc_match = re.search(r'Val Loss: [\d\.]+\s*\|\s*Val Acc: ([\d\.]+)%', content)
            if val_acc_match:
                metrics['Val Acc'] = float(val_acc_match.group(1))
            
            # Extract best validation accuracy
            best_val_match = re.search(r'Best validation accuracy: ([\d\.]+)%', content)
            if best_val_match:
                metrics['Best Val Acc'] = float(best_val_match.group(1))
        
        return metrics
    
    def extract_from_evaluation_log(self, log_file):
        """Extract metrics from NEW evaluation log format"""
        metrics = {}
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract test accuracy from NEW format
            # Looks for: "Test Accuracy: 9.44%"
            test_acc_match = re.search(r'Test Accuracy:\s*([\d\.]+)%', content)
            if test_acc_match:
                metrics['Clean Test Acc'] = float(test_acc_match.group(1))
                print(f"  Found test accuracy: {metrics['Clean Test Acc']}%")
            
            # Also look for the simple format (backward compatibility)
            simple_match = re.search(r'^Test Accuracy:\s*([\d\.]+)%', content, re.MULTILINE)
            if simple_match and 'Clean Test Acc' not in metrics:
                metrics['Clean Test Acc'] = float(simple_match.group(1))
            
            # Extract dataset info
            dataset_match = re.search(r'Dataset:\s*(.+)', content)
            if dataset_match:
                metrics['dataset_path'] = dataset_match.group(1)
            
            # Extract total samples
            samples_match = re.search(r'Total Samples:\s*(\d+)', content)
            if samples_match:
                metrics['total_samples'] = int(samples_match.group(1))
        
        return metrics
    
    def extract_from_model_checkpoint(self, model_path):
        """Extract metrics from model checkpoint"""
        metrics = {}
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'history' in checkpoint:
                    history = checkpoint['history']
                    if 'train_acc' in history and history['train_acc']:
                        metrics['Train Acc'] = history['train_acc'][-1]
                    if 'val_acc' in history and history['val_acc']:
                        metrics['Val Acc'] = history['val_acc'][-1]
                        metrics['Best Val Acc'] = max(history['val_acc'])
            except:
                pass
        
        return metrics
    
    def get_test_accuracy(self, eval_log_file):
        """Extract test accuracy from evaluation log"""
        if os.path.exists(eval_log_file):
            with open(eval_log_file, 'r') as f:
                content = f.read()
            
            # Look for test accuracy
            test_acc_match = re.search(r'Test Accuracy: ([\d\.]+)%', content)
            if test_acc_match:
                return float(test_acc_match.group(1))
        
        return None
    
    def get_adversarial_accuracy(self, adv_eval_log_file):
        """Extract adversarial test accuracy"""
        return self.get_test_accuracy(adv_eval_log_file)
    
    def add_experiment_result(self, experiment_name, config):
        """
        Add or update an experiment result
        
        Args:
            experiment_name: Name of the experiment (e.g., 'baseline', 'nonrobust_fgsm')
            config: Dictionary with experiment configuration and results
        """
        self.results[experiment_name] = config
        self.save_results()
    
    def auto_collect_experiment(self, experiment_name, model_dir, plots_dir):
        """
        Automatically collect results from experiment directories
        
        Args:
            experiment_name: Name for this experiment
            model_dir: Path to model outputs (e.g., 'outputs/models/baseline')
            plots_dir: Path to plot outputs (e.g., 'outputs/plots/baseline')
        """
        print(f"Collecting results for {experiment_name}...")
        
        # Default config
        config = {
            'experiment_name': experiment_name,
            'model_dir': str(model_dir),
            'plots_dir': str(plots_dir),
        }
        
        # Try to extract from training report
        train_report = os.path.join(plots_dir, f"{experiment_name}_training_report.txt")
        if os.path.exists(train_report):
            with open(train_report, 'r') as f:
                content = f.read()
                
            # Extract metrics using regex
            patterns = {
                'Final Train Acc': r'Final Training Accuracy:\s*([\d\.]+)%',
                'Final Val Acc': r'Final Validation Accuracy:\s*([\d\.]+)%',
                'Best Val Acc': r'Best Validation Accuracy:\s*([\d\.]+)%',
                'Test Acc': r'Test Accuracy:\s*([\d\.]+)%',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    config[key] = float(match.group(1))
        
        # Try to get test accuracy from evaluation
        #test_log = os.path.join(plots_dir, 'test_evaluation.log')
        #if os.path.exists(test_log):
        #    test_acc = self.get_test_accuracy(test_log)
        #    if test_acc:
        #        config['Clean Test Acc'] = test_acc

        # 1. Try to extract from NEW evaluation log
        eval_log = os.path.join(plots_dir, 'evaluation.log')
        if os.path.exists(eval_log):
            print(f"  Found evaluation log: {eval_log}")
            eval_metrics = self.extract_from_evaluation_log(eval_log)
            config.update(eval_metrics)
        else:
            print(f"  No evaluation.log found at: {eval_log}")
            # Try alternative names
            alt_logs = ['test_evaluation.log', 'eval.log', 'evaluation_results.log']
            for alt_log in alt_logs:
                alt_path = os.path.join(plots_dir, alt_log)
                if os.path.exists(alt_path):
                    print(f"  Found alternative log: {alt_path}")
                    eval_metrics = self.extract_from_evaluation_log(alt_path)
                    config.update(eval_metrics)
                    break
        
        # Try to get adversarial accuracy
        #adv_test_log = os.path.join(plots_dir, 'test_pgd_eps002', 'test_evaluation.log')
        #if os.path.exists(adv_test_log):
        #    adv_acc = self.get_adversarial_accuracy(adv_test_log)
        #    if adv_acc:
        #        config['Adv Test Acc'] = adv_acc
        #        
        #        # Calculate accuracy drop
        #        if 'Clean Test Acc' in config:
        #            config['Adv Acc Drop'] = config['Clean Test Acc'] - adv_acc


        # Look for ADVERSARIAL evaluation (PGD attacked test)
        # Check if there's a subdirectory with adversarial test results
        adv_dirs_to_check = [
            os.path.join(plots_dir, 'test_pgd_eps002'),
            os.path.join(plots_dir, 'adv_evaluation'),
            os.path.join(plots_dir, 'adversarial_test'),
            os.path.join(os.path.dirname(plots_dir), f'{experiment_name}_adv'),  # sibling directory
        ]
    
        for adv_dir in adv_dirs_to_check:
            if os.path.exists(adv_dir):
                print(f"  Checking adversarial directory: {adv_dir}")
                
                # Look for evaluation logs in this directory
                adv_logs_to_check = [
                    os.path.join(adv_dir, 'evaluation.log'),
                    os.path.join(adv_dir, 'test_evaluation.log'),
                    os.path.join(adv_dir, 'adv_evaluation.log'),
                ]
                
                for adv_log_path in adv_logs_to_check:
                    if os.path.exists(adv_log_path):
                        print(f"    Found adversarial log: {adv_log_path}")
                        adv_acc = self.extract_from_evaluation_log(adv_log_path).get('Clean Test Acc')
                        if adv_acc:
                            config['Adv Test Acc'] = adv_acc
                            
                            # Calculate accuracy drop
                            if 'Clean Test Acc' in config:
                                config['Adv Acc Drop'] = config['Clean Test Acc'] - adv_acc
                            break
                if 'Adv Test Acc' in config:
                    break
        
        # Extract from model checkpoint
        model_path = os.path.join(model_dir, 'best_model.pth')
        checkpoint_metrics = self.extract_from_model_checkpoint(model_path)
        config.update(checkpoint_metrics)
        
        # Add dataset info if available
        dataset_config = os.path.join(model_dir.replace('models', 'datasets'), 'dataset_config.json')
        if os.path.exists(dataset_config):
            with open(dataset_config, 'r') as f:
                dataset_info = json.load(f)
            config['dataset_info'] = dataset_info
        
        # Save the collected results
        self.add_experiment_result(experiment_name, config)
        
        print(f"  → Collected {len(config)} metrics")
        return config
    
    def save_results(self):
        """Save results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
    
    def get_dataframe(self):
        """Convert results to pandas DataFrame"""
        rows = []
        
        for exp_name, config in self.results.items():
            row = {'Experiment': exp_name}
            
            # Extract key metrics
            metrics = [
                'Train Acc', 'Val Acc', 'Best Val Acc', 
                'Clean Test Acc', 'Adv Test Acc', 'Adv Acc Drop'
            ]
            
            for metric in metrics:
                if metric in config:
                    row[metric] = config[metric]
                else:
                    row[metric] = None
            
            # Add dataset type if available
            if 'dataset_info' in config:
                row['Dataset Type'] = config['dataset_info'].get('dataset_type', 'clean')
                row['Attack Type'] = config['dataset_info'].get('attack_type', 'none')
                row['Epsilon'] = config['dataset_info'].get('epsilon', 0)
            else:
                row['Dataset Type'] = 'clean'
                row['Attack Type'] = 'none'
                row['Epsilon'] = 0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns
        col_order = ['Experiment', 'Dataset Type', 'Attack Type', 'Epsilon',
                    'Train Acc', 'Val Acc', 'Best Val Acc', 
                    'Clean Test Acc', 'Adv Test Acc', 'Adv Acc Drop']
        
        existing_cols = [c for c in col_order if c in df.columns]
        other_cols = [c for c in df.columns if c not in col_order]
        
        return df[existing_cols + other_cols]
    
    def generate_summary_table(self):
        """Generate a formatted summary table"""
        df = self.get_dataframe()
        
        # Format percentages
        percent_cols = ['Train Acc', 'Val Acc', 'Best Val Acc', 
                       'Clean Test Acc', 'Adv Test Acc', 'Adv Acc Drop']
        
        for col in percent_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        
        # Format epsilon
        if 'Epsilon' in df.columns:
            df['Epsilon'] = df['Epsilon'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "0")
        
        return df