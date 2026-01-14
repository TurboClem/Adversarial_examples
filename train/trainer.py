"""
Training and evaluation functions
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os

from config import *
from data_loader.dataset import EuroSatDataset


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, device=DEVICE):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader, criterion, optimizer, madry):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if madry:
                epsilon = madry['epsilon']
                alpha = madry['alpha']
                epsilon = epsilon.to(self.device) if torch.is_tensor(epsilon) else epsilon
                alpha = alpha.to(self.device) if torch.is_tensor(alpha) else alpha
                pgd_steps = madry['pgd_steps']
                inputs = self.pgd_attack_train(
                    inputs,
                    targets,
                    epsilon_pixel=epsilon,
                    alpha_pixel=alpha,
                    iterations=pgd_steps
                )

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE, madry=MADRY, save_model_path=SAVE_MODEL_PATH):
        """Train the model for multiple epochs"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,
        )
        
        best_val_acc = 0.0
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, madry)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                self.save_model(filename='best_model.pth', save_model_path=save_model_path)
                print(f"Saved best model with val_acc: {val_acc:.2f}%")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs with no improvement.")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return self.history
    
    def save_model(self, filename, save_model_path=SAVE_MODEL_PATH):
        """Save model checkpoint"""
        os.makedirs(save_model_path, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'model_type': type(self.model).__name__
        }, os.path.join(save_model_path, filename))
    
    def load_model(self, filename, save_model_path=SAVE_MODEL_PATH):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(save_model_path, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        return accuracy, all_preds, all_labels
    
    def pgd_attack_train(
        self,
        images,
        labels,
        epsilon_pixel,
        alpha_pixel,
        iterations
    ):
        self.model.eval()

        images = images.detach().to(self.device)
        labels = labels.to(self.device)

        epsilon = torch.tensor([epsilon_pixel / s for s in STD]).view(1,3,1,1).to(self.device)
        alpha = torch.tensor([alpha_pixel / s for s in STD]).view(1,3,1,1).to(self.device)
        
        # Random start
        delta = (torch.rand_like(images) * 2 - 1) * epsilon
        delta = torch.clamp(delta, -epsilon, epsilon)
        adv_images = images + delta
        adv_images = torch.clamp(adv_images, 0, 1)

        for _ in range(iterations):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            
            loss = nn.functional.cross_entropy(outputs, labels)

            grad = torch.autograd.grad(loss, adv_images)[0]

            adv_images = adv_images + alpha * torch.sign(grad)
            delta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

        return adv_images


def create_data_loaders(
    data_path=DATA_PATH_TRAIN,
    batch_size=BATCH_SIZE,
    mode="train"         # "train" or "eval"
):
    """
    Create PyTorch DataLoaders for EuroSAT

    Args:
        data_path (str or Path): Path to dataset (train_clean or test_clean)
        batch_size (int)
        mode (str): "train" -> returns train_loader, val_loader
                    "eval"  -> returns test_loader
    Returns:
        If mode=="train": (train_loader, val_loader, class_names)
        If mode=="eval" : (test_loader, class_names)
    """
    # Dataset
    train_flag = mode == "train"
    dataset = EuroSatDataset(data_path, train=train_flag)

    class_names = dataset.get_class_names()

    if mode == "train":
        # Split train/val
        total_len = len(dataset)
        val_size = int(0.3 * total_len)
        train_size = total_len - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )

        val_dataset.dataset.train = False

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        return train_loader, val_loader, class_names

    elif mode == "eval":
        # Test loader
        test_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        print(f"Test samples: {len(dataset)}")
        return test_loader, class_names

    else:
        raise ValueError("mode must be 'train' or 'eval'")
