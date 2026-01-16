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
        UPDATED for new directory structure with subdirectories
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
        
        # 1. FIRST: Look for CLEAN test evaluation (test_clean subdirectory)
        clean_eval_dir = os.path.join(plots_dir, 'test_clean')
        clean_eval_log = os.path.join(clean_eval_dir, 'evaluation.log')
        
        if os.path.exists(clean_eval_log):
            print(f"  Found clean evaluation log: {clean_eval_log}")
            clean_metrics = self.extract_from_evaluation_log(clean_eval_log)
            if 'Clean Test Acc' in clean_metrics:
                config['Clean Test Acc'] = clean_metrics['Clean Test Acc']
        else:
            # Fallback: look in main directory
            main_eval_log = os.path.join(plots_dir, 'evaluation.log')
            if os.path.exists(main_eval_log):
                print(f"  Found evaluation log in main dir: {main_eval_log}")
                clean_metrics = self.extract_from_evaluation_log(main_eval_log)
                if 'Clean Test Acc' in clean_metrics:
                    config['Clean Test Acc'] = clean_metrics['Clean Test Acc']
        
        # 2. SECOND: Look for ADVERSARIAL test evaluation
        # For non-robust experiments, adversarial test is on 'test_{model_name}'
        # For baseline/madry, adversarial test is on 'test_pgd_eps002'
        
        adv_acc_found = False
        
        # Option A: For non-robust models, look for test on adversarial version of themselves
        if 'nonrobust' in experiment_name or 'random' in experiment_name:
            adv_test_dir_name = f'test_{experiment_name}'
            adv_eval_dir = os.path.join(plots_dir, adv_test_dir_name)
            adv_eval_log = os.path.join(adv_eval_dir, 'evaluation.log')
            
            if os.path.exists(adv_eval_log):
                print(f"  Found self-adversarial evaluation: {adv_eval_log}")
                adv_metrics = self.extract_from_evaluation_log(adv_eval_log)
                if 'Clean Test Acc' in adv_metrics:
                    config['Adv Test Acc'] = adv_metrics['Clean Test Acc']
                    adv_acc_found = True
        
        # Option B: For all models, also look for PGD attacked test
        if not adv_acc_found:
            pgd_test_dir = os.path.join(plots_dir, 'test_pgd_eps002')
            pgd_eval_log = os.path.join(pgd_test_dir, 'evaluation.log')
            
            if os.path.exists(pgd_eval_log):
                print(f"  Found PGD adversarial evaluation: {pgd_eval_log}")
                pgd_metrics = self.extract_from_evaluation_log(pgd_eval_log)
                if 'Clean Test Acc' in pgd_metrics:
                    config['Adv Test Acc'] = pgd_metrics['Clean Test Acc']
                    adv_acc_found = True
        
        # Option C: Look for any subdirectory with 'test_' prefix
        if not adv_acc_found:
            import glob
            test_subdirs = glob.glob(os.path.join(plots_dir, 'test_*'))
            
            for test_dir in test_subdirs:
                if os.path.isdir(test_dir) and 'clean' not in test_dir.lower():
                    adv_log = os.path.join(test_dir, 'evaluation.log')
                    if os.path.exists(adv_log):
                        print(f"  Found test in {os.path.basename(test_dir)}: {adv_log}")
                        adv_metrics = self.extract_from_evaluation_log(adv_log)
                        if 'Clean Test Acc' in adv_metrics:
                            config['Adv Test Acc'] = adv_metrics['Clean Test Acc']
                            adv_acc_found = True
                            break
        
        # Calculate accuracy drop if both exist
        if 'Clean Test Acc' in config and 'Adv Test Acc' in config:
            config['Adv Acc Drop'] = config['Clean Test Acc'] - config['Adv Test Acc']
        
        # 3. Extract from model checkpoint for training metrics
        model_path = os.path.join(model_dir, 'best_model.pth')
        checkpoint_metrics = self.extract_from_model_checkpoint(model_path)
        config.update(checkpoint_metrics)
        
        # 4. Add dataset info if available
        # Look for dataset config in multiple locations
        possible_config_paths = [
            os.path.join(model_dir.replace('models', 'datasets'), 'dataset_config.json'),
            os.path.join('datasets/EuroSAT_RGB', f'train_{experiment_name}_config', 'dataset_config.json'),
            os.path.join('datasets/EuroSAT_RGB', f'{experiment_name}_config', 'dataset_config.json'),
        ]
        
        for config_path in possible_config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        dataset_info = json.load(f)
                    config['dataset_info'] = dataset_info
                    print(f"  Found dataset config: {config_path}")
                    break
                except Exception as e:
                    print(f"  Warning: Could not load config {config_path}: {e}")
        
        # 5. Set dataset type and attack type
        if 'dataset_info' in config:
            config['Dataset Type'] = config['dataset_info'].get('dataset_type', 'clean')
            config['Attack Type'] = config['dataset_info'].get('attack_type', 'none')
            config['Epsilon'] = config['dataset_info'].get('epsilon', 0)
        else:
            # Infer from experiment name
            if 'nonrobust' in experiment_name:
                config['Dataset Type'] = 'non_robust'
                config['Attack Type'] = 'fgsm' if 'fgsm' in experiment_name else 'pgd'
                config['Epsilon'] = 0.01  # Default from your code
            elif 'random' in experiment_name:
                config['Dataset Type'] = 'random_noise'
                config['Attack Type'] = 'random'
                config['Epsilon'] = 0.01
            elif 'madry' in experiment_name:
                config['Dataset Type'] = 'adversarial_training'
                config['Attack Type'] = 'pgd'
                if 'eps001' in experiment_name:
                    config['Epsilon'] = 0.01
                elif 'eps003' in experiment_name:
                    config['Epsilon'] = 0.03
                else:
                    config['Epsilon'] = 0.01  # Default
            else:
                config['Dataset Type'] = 'clean'
                config['Attack Type'] = 'none'
                config['Epsilon'] = 0.0
        
        # Save the collected results
        self.add_experiment_result(experiment_name, config)
        
        # Print summary
        summary = f"  → Collected {len(config)} metrics"
        if 'Clean Test Acc' in config:
            summary += f", Clean: {config['Clean Test Acc']}%"
        if 'Adv Test Acc' in config:
            summary += f", Adv: {config['Adv Test Acc']}%"
        print(summary)
        
        return config

    def save_results(self):
        """Save results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
    
    def get_dataframe(self):
        """Convert results to pandas DataFrame - IMPROVED"""
        rows = []
        
        for exp_name, config in self.results.items():
            row = {'Experiment': exp_name}
            
            # Extract key metrics with better handling
            metrics_mapping = {
                'Train Acc': ['Train Acc', 'Final Train Acc'],
                'Val Acc': ['Val Acc', 'Final Val Acc'],
                'Best Val Acc': ['Best Val Acc'],
                'Clean Test Acc': ['Clean Test Acc', 'Test Acc'],
                'Adv Test Acc': ['Adv Test Acc'],
                'Adv Acc Drop': ['Adv Acc Drop'],
            }
            
            for display_name, possible_keys in metrics_mapping.items():
                value = None
                for key in possible_keys:
                    if key in config and config[key] is not None:
                        # Handle different formats
                        if isinstance(config[key], (int, float)):
                            value = float(config[key])
                        elif isinstance(config[key], str):
                            # Try to extract number from string like "9.44%"
                            num_match = re.search(r'([\d\.]+)', config[key])
                            if num_match:
                                value = float(num_match.group(1))
                        if value is not None:
                            break
                row[display_name] = value
            
            # Add dataset info - use direct config if available
            row['Dataset Type'] = config.get('Dataset Type', 
                                            config.get('dataset_info', {}).get('dataset_type', 'clean'))
            row['Attack Type'] = config.get('Attack Type',
                                        config.get('dataset_info', {}).get('attack_type', 'none'))
            row['Epsilon'] = config.get('Epsilon',
                                    config.get('dataset_info', {}).get('epsilon', 0))
            
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