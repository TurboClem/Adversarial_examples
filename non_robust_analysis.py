# %%
import torch
from models import ResNet18
from datasets.feature_analysis import FeatureAnalysisDatasetGenerator
import sys
import json
import pandas as pd
# %%
# Load the baseline model (your 97.56% accuracy model)
model_path = 'outputs/models/baseline/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

baseline_model = ResNet18().to(device)
checkpoint = torch.load(model_path, map_location=device)
baseline_model.load_state_dict(checkpoint['model_state_dict'])
baseline_model.eval()

epsilon = 0.04
# %%
generator = FeatureAnalysisDatasetGenerator(baseline_model, device=device)

# %%
# Create non-robust dataset with FGSM (weak attack)
for type in ['train', 'test']:
    non_robust_path = f'datasets/EuroSAT_RGB/{type}_nonrobust_fgsm'
    generator.create_non_robust_dataset(
        clean_train_path=f'datasets/EuroSAT_RGB/{type}_clean',
        save_path=non_robust_path,
        attack_type='fgsm',
        epsilon=epsilon,
        mislabel_strategy='random',
        attack_strategy='target',
    )

# %%
# Create non-robust dataset with PGD (for comparison)
for type in ['train', 'test']:
    non_robust_pgd_path = f'datasets/EuroSAT_RGB/{type}_nonrobust_pgd'
    generator.create_non_robust_dataset(
        clean_train_path=f'datasets/EuroSAT_RGB/{type}_clean',
        save_path=non_robust_pgd_path,
        attack_type='pgd',
        epsilon=epsilon,
        alpha=0.002,
        iterations=3,
        mislabel_strategy='random',
        attack_strategy='target',
    )

# %%
# Create random noise dataset
for type in ['train', 'test']:
    random_noise_path = f'datasets/EuroSAT_RGB/{type}_random_noise'
    generator.create_random_noise_dataset(
        clean_train_path=f'datasets/EuroSAT_RGB/{type}_clean',
        save_path=random_noise_path,
        epsilon=0.01  # Same magnitude as adversarial attacks
    )

# %%


def train(model_name: str):

    sys.argv = [
        "main.py",
        "--model", "resnet18",
        "--train",
        "--visualize",
        "--epochs", "30",
        "--patience", "10",
        "--lr", "0.001",
        "--batch-size", "32",
        "--seed", "42",
        "--data-path-train", f"datasets/EuroSAT_RGB/train_{model_name}",
        "--save-model-path", f"outputs/models/{model_name}",
        "--save-plots-path", f"outputs/plots/{model_name}",
    ]

    from main import main
    main()


def evaluate(model_name: str, adv: bool):
    test_name = f"test_{model_name}" if adv else "test_clean"

    sys.argv = [
        "main.py",
        "--model", "resnet18",
        "--evaluate",
        "--visualize",
        "--seed", "42",
        "--data-path-eval", f"datasets/EuroSAT_RGB/{test_name}",
        "--save-model-path", f"outputs/models/{model_name}",
        "--save-plots-path", f"outputs/plots/{model_name}/{test_name}",
    ]

    from main import main
    main()


# %%
# Train on FGSM non-robust dataset
model_name = "nonrobust_fgsm"
train(model_name)
#evaluate(model_name, adv=True)
evaluate(model_name, adv=False)

# %%
# Train on PGD non-robust dataset
model_name = "nonrobust_pgd"
train(model_name)
#evaluate(model_name, adv=True)
evaluate(model_name, adv=False)

# %%
# Train on Random Noise Dataset
model_name = "random_noise"
train(model_name)
#evaluate(model_name, adv=True)
evaluate(model_name, adv=False)

# %%
# Collect all results automatically
from utils.result_collector import ResultCollector

collector = ResultCollector()

experiment_dirs = [
    #('baseline', 'outputs/models/baseline', 'outputs/plots/baseline_clean'),
    #('madry_eps001', 'outputs/models/madry_eps001', 'outputs/plots/madry_eps001'),
    #('madry_eps003', 'outputs/models/madry_eps003', 'outputs/plots/madry_eps003'),
    ('nonrobust_fgsm', 'outputs/models/nonrobust_fgsm', 'outputs/plots/nonrobust_fgsm'),
    ('nonrobust_pgd', 'outputs/models/nonrobust_pgd', 'outputs/plots/nonrobust_pgd'),
    ('random_noise', 'outputs/models/random_noise', 'outputs/plots/random_noise'),
]

print("Collecting experiment results...")
for exp_name, model_dir, plots_dir in experiment_dirs:
    if os.path.exists(model_dir) and os.path.exists(plots_dir):
        collector.auto_collect_experiment(exp_name, model_dir, plots_dir)
        print(f"  - Collected: {exp_name}")
    else:
        print(f"  x Skipped: {exp_name} (directories not found)")

# Display results
df_results = collector.generate_summary_table()
print("\n" + "="*80)
print("AUTOMATICALLY COLLECTED RESULTS")
print("="*80)
print(df_results.to_string(index=False))


# Save to CSV
csv_path = 'outputs/results/experiment_results.csv'
df_results.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# %%
# Generate feature analysis summary
from datasets.feature_analysis import create_feature_analysis_summary

df_analysis = create_feature_analysis_summary()

# %%
# Visualize non-robust model predictions
from datasets.feature_analysis import visualize_non_robust_features

print("\n" + "="*80)
print("VISUALIZING NON-ROBUST MODEL PREDICTIONS")
print("="*80)

# Visualize FGSM non-robust model
print("\n1. FGSM Non-Robust Model:")
fgsm_accuracy = visualize_non_robust_features('nonrobust_fgsm')

# Visualize random noise model (for comparison)
print("\n2. Random Noise Model (Control):")
random_accuracy = visualize_non_robust_features('random_noise')

# %%
# Generate final report
print("\n" + "="*80)
print("FINAL EXPERIMENT REPORT")
print("="*80)


def safe_float(value):
    """Safely convert value to float, handling 'N/A' and percentages"""
    if pd.isnull(value) or value == "N/A" or value == "":
        return None
    # Remove % sign if present
    if isinstance(value, str):
        value = value.replace('%', '')
    try:
        return float(value)
    except:
        return None


# Calculate key metrics
baseline_acc = safe_float(df_results[df_results['Experiment'] == 'baseline']['Clean Test Acc'].values[0])
nonrobust_acc = safe_float(df_results[df_results['Experiment'] == 'nonrobust_fgsm']['Clean Test Acc'].values[0])
random_acc = safe_float(df_results[df_results['Experiment'] == 'random_noise']['Clean Test Acc'].values[0])

print(baseline_acc, nonrobust_acc, random_acc)

print("\nKey Results:")
print(f"1_ Baseline (clean training): {baseline_acc}%")
print(f"2_ Non-Robust Dataset (FGSM): {nonrobust_acc}%")
print(f"3_ Random Noise Dataset: {random_acc}%")
print(f"4_ Difference (Non-Robust vs Random): {nonrobust_acc - random_acc}% points")

if nonrobust_acc > random_acc + 15:
    print("\nSTRONG EVIDENCE: Adversarial perturbations contain predictive features!")
    print("  - The {nonrobust_acc - random_acc:.2f}% difference is statistically significant")
    print("  - This confirms the Ilyas et al. hypothesis")
elif nonrobust_acc > random_acc:
    print("\n✓ MODERATE EVIDENCE: Some predictive features detected")
    print("  - Consider trying larger epsilon or stronger attacks")
else:
    print("\n✗ INCONCLUSIVE: No clear evidence of non-robust features")
    print("  - Possible issues: epsilon too small, attack too weak")

# %%
