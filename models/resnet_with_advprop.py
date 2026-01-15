# models/resnet_with_advprop.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES


class AdvPropResNet(nn.Module):
    """ResNet with auxiliary BatchNorm layers for AdvProp"""
    
    def __init__(self, block, num_blocks, num_classes=NUM_CLASSES):
        super(AdvPropResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_main = nn.BatchNorm2d(64)      # Main BN for clean images
        self.bn1_aux = nn.BatchNorm2d(64)       # Auxiliary BN for adversarial images
        
        # Residual blocks - need to create with dual BNs
        self.layer1 = self._make_layer_with_advprop(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer_with_advprop(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer_with_advprop(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_with_advprop(block, 512, num_blocks[3], stride=2)
        
        # Classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer_with_advprop(self, block, out_channels, num_blocks, stride):
        """Create layers with both main and auxiliary BNs"""
        layers = []
        for i in range(num_blocks):
            # Create block with dual BN support
            layers.append(AdvPropBlock(block, self.in_channels, out_channels, 
                                     stride if i == 0 else 1))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, use_aux_bn=False):
        """Forward pass with option to use auxiliary BNs"""
        # Choose which BN to use
        bn1 = self.bn1_aux if use_aux_bn else self.bn1_main
        
        out = F.relu(bn1(self.conv1(x)))
        
        # Forward through all layers with the chosen BN mode
        for layer in self.layer1:
            out = layer(out, use_aux_bn)
        for layer in self.layer2:
            out = layer(out, use_aux_bn)
        for layer in self.layer3:
            out = layer(out, use_aux_bn)
        for layer in self.layer4:
            out = layer(out, use_aux_bn)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
    
    def train_forward(self, x_clean, x_adv):
        """Special forward for AdvProp training"""
        # Clean images through main BNs
        out_clean = self.forward(x_clean, use_aux_bn=False)
        
        # Adversarial images through auxiliary BNs
        out_adv = self.forward(x_adv, use_aux_bn=True)
        
        return out_clean, out_adv


class AdvPropBlock(nn.Module):
    """Basic block with dual BatchNorm for AdvProp"""
    
    def __init__(self, block_class, in_channels, out_channels, stride=1):
        super(AdvPropBlock, self).__init__()
        
        # Create the base block
        self.block = block_class(in_channels, out_channels, stride)
        
        # CORRECTION ICI : Ne pas modifier pendant l'itération
        self._replace_bns_with_dual()
    
    def _replace_bns_with_dual(self):
        """Replace all BatchNorm layers with dual versions - CORRIGÉ"""
        # Premièrement, collecter tous les BN à remplacer
        bn_to_replace = []
        for name, module in self.block.named_children():
            if isinstance(module, nn.BatchNorm2d):
                bn_to_replace.append((name, module))
        
        # Ensuite, les remplacer
        for name, module in bn_to_replace:
            # Créer les dual BNs
            setattr(self.block, f"{name}_main", nn.BatchNorm2d(module.num_features))
            setattr(self.block, f"{name}_aux", nn.BatchNorm2d(module.num_features))
            # Supprimer l'original
            delattr(self.block, name)
    
    def forward(self, x, use_aux_bn=False):
        """Forward with selected BN"""
        residual = x
        
        # CORRECTION : Gérer le shortcut correctement
        # Le shortcut est géré par le block lui-même dans ResNet
        if hasattr(self.block, 'shortcut'):
            # Pour les blocs avec changement de dimension
            shortcut = self.block.shortcut(x)
        else:
            # Pour les blocs sans changement de dimension
            shortcut = x
        
        # First conv + BN
        conv1_out = self.block.conv1(x)
        # CORRECTION : Vérifier si le BN existe
        bn1_main_name = 'bn1_main' if hasattr(self.block, 'bn1_main') else 'bn1'
        bn1_aux_name = 'bn1_aux' if hasattr(self.block, 'bn1_aux') else 'bn1'
        
        if use_aux_bn:
            bn1 = getattr(self.block, bn1_aux_name, getattr(self.block, 'bn1', None))
        else:
            bn1 = getattr(self.block, bn1_main_name, getattr(self.block, 'bn1', None))
        
        if bn1 is None:
            # Si pas de BN, utiliser juste conv
            out = F.relu(conv1_out)
        else:
            out = F.relu(bn1(conv1_out))
        
        # Second conv + BN
        out = self.block.conv2(out)
        bn2_main_name = 'bn2_main' if hasattr(self.block, 'bn2_main') else 'bn2'
        bn2_aux_name = 'bn2_aux' if hasattr(self.block, 'bn2_aux') else 'bn2'
        
        if use_aux_bn:
            bn2 = getattr(self.block, bn2_aux_name, getattr(self.block, 'bn2', None))
        else:
            bn2 = getattr(self.block, bn2_main_name, getattr(self.block, 'bn2', None))
        
        if bn2 is not None:
            out = bn2(out)
        
        # Add residual
        out += shortcut
        out = F.relu(out)
        
        return out


def ResNet18AdvProp(num_classes=NUM_CLASSES):
    """Create ResNet-18 with AdvProp support"""
    # CORRECTION : Import correct
    try:
        from .resnet import BasicBlock
        block_class = BasicBlock
    except ImportError:
        # Fallback : définir BasicBlock localement
        class BasicBlock(nn.Module):
            expansion = 1
            
            def __init__(self, in_channels, out_channels, stride=1):
                super(BasicBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                      stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                      stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != self.expansion * out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, self.expansion * out_channels,
                                 kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion * out_channels)
                    )
            
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out
        
        block_class = BasicBlock
    
    return AdvPropResNet(block_class, [2, 2, 2, 2], num_classes=num_classes)