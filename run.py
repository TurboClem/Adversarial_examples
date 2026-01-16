model_path = 'outputs/models/baseline/best_model.pth'
import sys

sys.argv = [
        "main.py",
        "--model", "resnet18",
        "--evaluate",
        "--visualize",
        "--seed", "42",
        "--data-path-eval", f"datasets/EuroSAT_RGB/test_clean",
        "--save-model-path", 'outputs/models/baseline',  # f"outputs/models/{model_name}",
        "--save-plots-path", f"outputs/plots/baseline/baseline_test",
    ]

from main import main
main()