"""
AdvProp Trainer - Implementation of "Adversarial Examples Improve Image Recognition"
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config import *
from attacks.pgd import PGD
import torch.nn.functional as F  # <-- Ajouter


class AdvPropTrainer:
    """Trainer for AdvProp (Adversarial Propagation)"""
    
    def __init__(self, model, epsilon=0.03, alpha=0.01, 
                 iterations=7, device=DEVICE):
        self.model = model.to(device)
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc_clean': [],
            'train_acc_adv': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"AdvProp initialized with epsilon={epsilon}, alpha={alpha}")
    
    def generate_adversarial_batch(self, images, labels):
        """Generate adversarial examples for AdvProp"""
        # IMPORTANT: AdvProp paper uses DIFFERENT epsilon for training
        # They typically use smaller epsilon during training
        self.model.eval()
        
        # Convert epsilon to tensor format for normalization
        epsilon_tensor = torch.tensor([self.epsilon / s for s in STD])\
            .view(1, 3, 1, 1).to(self.device)
        alpha_tensor = torch.tensor([self.alpha / s for s in STD])\
            .view(1, 3, 1, 1).to(self.device)
        
        # Random start (important for PGD)
        delta = (torch.rand_like(images) * 2 - 1) * epsilon_tensor
        delta = torch.clamp(delta, -epsilon_tensor, epsilon_tensor)
        adv_images = images + delta
        adv_images = torch.clamp(adv_images, 0, 1)
        
        # PGD iterations
        for _ in range(self.iterations):
            adv_images.requires_grad = True
            
            # CRITICAL: Use auxiliary BNs for attack generation
            outputs = self.model(adv_images, use_aux_bn=True)
            loss = F.cross_entropy(outputs, labels)
            
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images.detach() + alpha_tensor * torch.sign(grad.detach())
            
            # Project back to epsilon ball
            delta = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
            adv_images = torch.clamp(images + delta, 0, 1).detach()
        
        return adv_images
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch with AdvProp"""
        self.model.train()
        total_loss = 0.0
        correct_clean = 0
        correct_adv = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='AdvProp Training', leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Generate adversarial examples
            adv_inputs = self.generate_adversarial_batch(inputs, targets)
            
            # IMPORTANT: AdvProp requires TWO forward passes
            # Clean images through main BNs
            outputs_clean = self.model(inputs, use_aux_bn=False)
            
            # Adversarial images through auxiliary BNs
            outputs_adv = self.model(adv_inputs, use_aux_bn=True)
            
            # Calculate losses
            loss_clean = criterion(outputs_clean, targets)
            loss_adv = criterion(outputs_adv, targets)
            
            # Combined loss (equal weighting)
            loss = loss_clean + loss_adv
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            _, preds_clean = outputs_clean.max(1)
            _, preds_adv = outputs_adv.max(1)
            
            correct_clean += preds_clean.eq(targets).sum().item()
            correct_adv += preds_adv.eq(targets).sum().item()
            total += targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc_clean': 100. * correct_clean / total,
                'acc_adv': 100. * correct_adv / total
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc_clean = 100. * correct_clean / total
        epoch_acc_adv = 100. * correct_adv / total
        
        return epoch_loss, epoch_acc_clean, epoch_acc_adv
    
    def validate(self, val_loader, criterion):
        """Validate on clean images (using main BNs only)"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Use main BNs for validation
                outputs = self.model(inputs, use_aux_bn=False)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE,
              patience=PATIENCE, save_model_path=SAVE_MODEL_PATH):
        """Train model with AdvProp - aligned with ModelTrainer API"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        best_val_acc = 0.0
        epochs_no_improve = 0
        
        print(f"\nStarting AdvProp Training")
        print(f"Epochs: {epochs}, Epsilon: {self.epsilon}")
        print("-" * 60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train with AdvProp
            train_loss, train_acc_clean, train_acc_adv = self.train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc_clean'].append(train_acc_clean)
            self.history['train_acc_adv'].append(train_acc_adv)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc (Clean): {train_acc_clean:.2f}%")
            print(f"Train Acc (Adv): {train_acc_adv:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                self.save_model('best_model_advprop.pth', save_model_path)
                print(f"Saved best model with val_acc: {val_acc:.2f}%")
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nAdvProp training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate model on test set - aligned with ModelTrainer API"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Use main BNs for evaluation
                outputs = self.model(inputs, use_aux_bn=False)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        return accuracy, all_preds, all_labels
    
    def save_model(self, filename, save_model_path=SAVE_MODEL_PATH):
        """Save model checkpoint"""
        os.makedirs(save_model_path, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'iterations': self.iterations,
            'model_type': 'ResNet18AdvProp'
        }, os.path.join(save_model_path, filename))
    
    def load_model(self, filename, save_model_path=SAVE_MODEL_PATH):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(save_model_path, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint