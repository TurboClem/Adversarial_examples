"""
Simple CNN model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES


class SimpleCNN(nn.Module):
    """
    A simple CNN model for image classification
    Implemented from scratch as per project requirements
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Calculate flattened size
        # After 4 pooling layers of size 2: 64 -> 32 -> 16 -> 8 -> 4
        self.flattened_size = 256 * 4 * 4

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Fourth conv block
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

    def get_features(self, x):
        """Extract features from intermediate layers (useful for adversarial analysis)"""
        features = []

        x = F.relu(self.bn1(self.conv1(x)))
        features.append(x)
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        features.append(x)
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        features.append(x)
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        features.append(x)
        x = self.pool(x)

        return features


def create_simple_cnn(num_classes=NUM_CLASSES):
    """Factory function to create SimpleCNN"""
    return SimpleCNN(num_classes=num_classes)
