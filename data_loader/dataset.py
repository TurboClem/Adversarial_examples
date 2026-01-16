"""
Custom dataset loader for EuroSat dataset
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import IMG_SIZE, MEAN, STD


class EuroSatDataset(Dataset):
    """EuroSat Land Cover Dataset loader"""

    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with all the class folders
            transform (callable, optional): Transform to be applied
            train (bool): If True, apply training transforms
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform or self._get_default_transform()

        # Get all classes
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        # Load image paths and labels
        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])

        print(f"Loaded {len(self.image_paths)} images from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def _get_default_transform(self):
        """Get default transforms for training or testing"""
        if self.train:
            return transforms.Compose(
                [
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        """Get list of class names"""
        return self.classes
