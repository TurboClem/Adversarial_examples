# train/advprop_trainer.py
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


class AdvPropTrainer:
    """Trainer for AdvProp (Adversarial Propagation)"""
    
    def __init__(self, model, epsilon=0.03, alpha=0.01, 
                 iterations=7, device=DEVICE):
        self.model = model.to(device)
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        
        # Attack for generating adversarial examples
        self.attack = PGD(
            model, 
            epsilon=epsilon,
            alpha=alpha,
            iterations=iterations,
            random_start=True
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc_clean': [],
            'train_acc_adv': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def generate_adversarial_batch(self, images, labels):
        """Generate adversarial examples using auxiliary BNs"""
        # Important: Use model in eval mode with auxiliary BNs for attack generation
    
        original_training = self.model.training
        original_use_aux = getattr(self.model, '_use_aux_bn', False)

        self.model.eval()
        
        # Generate attack using auxiliary BNs
        #with torch.no_grad():
        adv_images = self.attack.attack(images, labels)
        if original_training:
            self.model.train()
        if original_use_aux:
        # Réactive aux BN si c'était activé
            pass

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
            
            # Generate adversarial examples using auxiliary BNs
            adv_inputs = self.generate_adversarial_batch(inputs, targets)
            
            # Forward pass with AdvProp
            outputs_clean, outputs_adv = self.model.train_forward(inputs, adv_inputs)
            
            # Calculate losses
            loss_clean = criterion(outputs_clean, targets)
            loss_adv = criterion(outputs_adv, targets)
            
            # Combined loss (equal weighting as in paper)
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
                
                # Use main BNs for validation (clean images)
                outputs = self.model(inputs, use_aux_bn=False)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
        """Train model with AdvProp"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0.0
        
        print(f"\nStarting AdvProp Training")
        print(f"Epsilon: {self.epsilon}, Alpha: {self.alpha}, Iterations: {self.iterations}")
        print("-" * 60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train with AdvProp
            train_loss, train_acc_clean, train_acc_adv = self.train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
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
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model_advprop.pth')
                print(f"  → Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print(f"\nAdvProp training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.history

    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs, use_aux_bn=False)  # Use main BNs for evaluation
                _, predicted = outputs.max(1)
            
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
    
        accuracy = 100. * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        return accuracy, all_preds, all_labels
    
    def save_model(self, filename):
        """Save model checkpoint"""
        os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'iterations': self.iterations
        }, os.path.join(SAVE_MODEL_PATH, filename))
    
    def load_model(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(SAVE_MODEL_PATH, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint
    
    def evaluate_robustness(self, test_loader, attack_strength=0.03):
        """Evaluate model robustness against different attacks"""
        from attacks.fgsm import FGSM
        from attacks.pgd import PGD
        
        self.model.eval()
        
        # Test on clean images
        clean_acc, _, _ = self._evaluate_attack(test_loader, attack=None)
        
        # Test against FGSM
        fgsm = FGSM(self.model, epsilon=attack_strength)
        fgsm_acc, _, _ = self._evaluate_attack(test_loader, attack=fgsm)
        
        # Test against PGD
        pgd = PGD(self.model, epsilon=attack_strength, alpha=attack_strength/4, iterations=10)
        pgd_acc, _, _ = self._evaluate_attack(test_loader, attack=pgd)
        
        return {
            'clean_accuracy': clean_acc,
            'fgsm_accuracy': fgsm_acc,
            'pgd_accuracy': pgd_acc,
            'fgsm_drop': clean_acc - fgsm_acc,
            'pgd_drop': clean_acc - pgd_acc
        }
    
    def _evaluate_attack(self, test_loader, attack=None):
        """Helper function for evaluation"""
        correct = 0
        total = 0
        
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if attack is not None:
                inputs = attack.attack(inputs, targets)
            
            outputs = self.model(inputs, use_aux_bn=False)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy, correct, total