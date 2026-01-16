# compute_stats.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_loader.dataset import EuroSatDataset

def compute_stats():
    train_path = "./datasets/EuroSAT_RGB/train_clean"
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # Loads only train set
    dataset = EuroSatDataset(
        root_dir=train_path,
        transform=transform,
        train=True
    )
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    mean = 0.
    std = 0.
    nb_samples = 0.
    
    for data, _ in dataloader:
        # data: [batch, 3, H, W]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    
    print(f"MEAN = {mean.tolist()}")
    print(f"STD = {std.tolist()}")
    
    return mean.tolist(), std.tolist()


def save_statistics_to_config(mean, std, config_file_path="config.py"):
    """
    Saves statistics into config file.
    
    Args:
        mean: Mean list [R, G, B]
        std: Standard errors list [R, G, B]
        config_file_path: Path to config.py
    """
    
    with open(config_file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.strip().startswith("MEAN ="):
            new_lines.append(f"MEAN = {mean}  # Computed on train set\n")
        elif line.strip().startswith("STD ="):
            new_lines.append(f"STD = {std}  # Computed on train set\n")
        else:
            new_lines.append(line)
    
    # Écrire le fichier mis à jour
    with open(config_file_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Statistiques sauvegardées dans {config_file_path}")


if __name__ == "__main__":
    mean, std = compute_stats()
    save_statistics_to_config(mean, std, config_file_path="config.py")
