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
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
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
    
    def train(self, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
        """Train the model for multiple epochs"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,
        )
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
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
                self.save_model('best_model.pth')
                print(f"Saved best model with val_acc: {val_acc:.2f}%")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return self.history
    
    def save_model(self, filename):
        """Save model checkpoint"""
        os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'model_type': type(self.model).__name__
        }, os.path.join(SAVE_MODEL_PATH, filename))
    
    def load_model(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(SAVE_MODEL_PATH, filename))
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


def create_data_loaders(data_path=DATA_PATH, batch_size=BATCH_SIZE):
    """Create train, validation, and test data loaders"""
    # Create full dataset
    full_dataset = EuroSatDataset(data_path, train=True)
    
    # Split into train, val, test
    train_size = int(TRAIN_TEST_SPLIT * len(full_dataset))
    val_size = int(0.5 * (len(full_dataset) - train_size))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Update val and test datasets to use test transforms
    val_dataset.dataset.train = False
    test_dataset.dataset.train = False
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, full_dataset.get_class_names()