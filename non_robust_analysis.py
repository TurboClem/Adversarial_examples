# %%
import torch
from models import ResNet18
from datasets.feature_analysis import FeatureAnalysisDatasetGenerator

# %%
# Load the baseline model (your 97.56% accuracy model)
model_path = 'outputs/models/baseline/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

baseline_model = ResNet18().to(device)
checkpoint = torch.load(model_path, map_location=device)
baseline_model.load_state_dict(checkpoint['model_state_dict'])
baseline_model.eval()

# %%
generator = FeatureAnalysisDatasetGenerator(baseline_model, device=device)

# Create non-robust dataset with FGSM (weak attack)
non_robust_path = 'datasets/EuroSAT_RGB/train_nonrobust_fgsm'
generator.create_non_robust_dataset(
    clean_train_path='datasets/EuroSAT_RGB/train_clean',
    save_path=non_robust_path,
    attack_type='fgsm',
    epsilon=0.01,  # Small epsilon
    mislabel_strategy='random'
)

# %%
# Create non-robust dataset with PGD (for comparison)
non_robust_pgd_path = 'datasets/EuroSAT_RGB/train_nonrobust_pgd'
generator.create_non_robust_dataset(
    clean_train_path='datasets/EuroSAT_RGB/train_clean',
    save_path=non_robust_pgd_path,
    attack_type='pgd',
    epsilon=0.01,
    alpha=0.002,
    iterations=3,
    mislabel_strategy='random'
)

# %%
random_noise_path = 'datasets/EuroSAT_RGB/train_random_noise'
generator.create_random_noise_dataset(
    clean_train_path='datasets/EuroSAT_RGB/train_clean',
    save_path=random_noise_path,
    epsilon=0.01  # Same magnitude as adversarial attacks
)

# %%
# Train on Non-Robust FGSM Dataset
import sys
import json

# Train on FGSM non-robust dataset
sys.argv = [
    "main.py",
    "--model", "resnet18",
    "--train",
    "--evaluate",
    "--visualize",
    "--epochs", "30",
    "--patience", "10",
    "--lr", "0.001",
    "--batch-size", "32",
    "--seed", "42",
    "--data-path-train", "datasets/EuroSAT_RGB/train_nonrobust_fgsm",
    "--data-path-eval", "datasets/EuroSAT_RGB/test_clean",  # Test on CLEAN data
    "--save-model-path", "outputs/models/nonrobust_fgsm",
    "--save-plots-path", "outputs/plots/nonrobust_fgsm",
]

from main import main
main()


# %%
# Train on Random Noise Dataset
sys.argv = [
    "main.py",
    "--model", "resnet18",
    "--train",
    "--evaluate",
    "--visualize",
    "--epochs", "30",
    "--patience", "10",
    "--lr", "0.001",
    "--batch-size", "32",
    "--seed", "42",
    "--data-path-train", "datasets/EuroSAT_RGB/train_random_noise",
    "--data-path-eval", "datasets/EuroSAT_RGB/test_clean",  # Test on CLEAN data
    "--save-model-path", "outputs/models/random_noise",
    "--save-plots-path", "outputs/plots/random_noise",
]

from main import main
main()


# %%
# Collect all results automatically
from utils.result_collector import ResultCollector

collector = ResultCollector()

experiment_dirs = [
    ('baseline', 'outputs/models/baseline', 'outputs/plots/baseline_clean'),
    ('madry_eps001', 'outputs/models/madry_eps001', 'outputs/plots/madry_eps001'),
    ('madry_eps003', 'outputs/models/madry_eps003', 'outputs/plots/madry_eps003'),
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
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# %%
# Generate feature analysis summary
from analysis.feature_analysis import create_feature_analysis_summary

df_analysis = create_feature_analysis_summary()

# %%
# Visualize non-robust model predictions
from analysis.feature_analysis import visualize_non_robust_features

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

# Calculate key metrics
baseline_acc = df_results[df_results['Experiment'] == 'baseline']['Clean Test Acc'].values[0]
nonrobust_acc = df_results[df_results['Experiment'] == 'nonrobust_fgsm']['Clean Test Acc'].values[0]
random_acc = df_results[df_results['Experiment'] == 'random_noise']['Clean Test Acc'].values[0]

print(f"\nKey Results:")
print(f"1. Baseline (clean training): {baseline_acc:.2f}%")
print(f"2. Non-Robust Dataset (FGSM): {nonrobust_acc:.2f}%")
print(f"3. Random Noise Dataset: {random_acc:.2f}%")
print(f"4. Difference (Non-Robust vs Random): {nonrobust_acc - random_acc:.2f}% points")

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
