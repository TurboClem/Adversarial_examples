import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES


class AdvPropResNet(nn.Module):
    """ResNet-18 with AdvProp support - SIMPLE WORKING VERSION"""
    
    def __init__(self, num_classes=NUM_CLASSES):
        super(AdvPropResNet, self).__init__()
        
        # Initial layers with dual BNs
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_main = nn.BatchNorm2d(64)
        self.bn1_aux = nn.BatchNorm2d(64)
        
        # Layer 1
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        
        # Layer 2
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        
        # Layer 3
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Layer 4
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple blocks"""
        layers = []
        
        # First block with possible stride
        layers.append(AdvPropBasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(AdvPropBasicBlock(out_channels, out_channels, stride=1))
        
        return nn.ModuleList(layers)
    
    def forward(self, x, use_aux_bn=False):
        """Forward pass"""
        # Initial conv + BN
        bn1 = self.bn1_aux if use_aux_bn else self.bn1_main
        x = F.relu(bn1(self.conv1(x)))
        
        # Through all layers
        for block in self.layer1:
            x = block(x, use_aux_bn)
        for block in self.layer2:
            x = block(x, use_aux_bn)
        for block in self.layer3:
            x = block(x, use_aux_bn)
        for block in self.layer4:
            x = block(x, use_aux_bn)
        
        # Classifier
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class AdvPropBasicBlock(nn.Module):
    """Basic Block with dual BatchNorm for AdvProp"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(AdvPropBasicBlock, self).__init__()
        
        # Main convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        # Dual BatchNorm layers
        self.bn1_main = nn.BatchNorm2d(out_channels)
        self.bn1_aux = nn.BatchNorm2d(out_channels)
        self.bn2_main = nn.BatchNorm2d(out_channels)
        self.bn2_aux = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x, use_aux_bn=False):
        """Forward with selected BN"""
        residual = x
        
        # Choose which BNs to use
        if use_aux_bn:
            bn1 = self.bn1_aux
            bn2 = self.bn2_aux
        else:
            bn1 = self.bn1_main
            bn2 = self.bn2_main
        
        # First convolution
        out = self.conv1(x)
        out = bn1(out)
        out = F.relu(out)
        
        # Second convolution
        out = self.conv2(out)
        out = bn2(out)
        
        # Shortcut
        residual = self.shortcut(x)
        
        # Residual connection
        out += residual
        out = F.relu(out)
        
        return out


def ResNet18AdvProp(num_classes=NUM_CLASSES):
    """Create ResNet-18 with AdvProp support"""
    return AdvPropResNet(num_classes=num_classes)