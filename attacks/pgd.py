"""
PGD Attack: Projected Gradient Descent
Paper: "Towards Deep Learning Models Resistant to Adversarial Attacks" (Madry et al., 2018)

Key idea: Multi-step attack with projection
Iterative version of FGSM with random start
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attack import BaseAttack


class PGD(BaseAttack):
    
    def __init__(self, model, epsilon=0.03, alpha=0.01, 
                 iterations=10, random_start=True, 
                 targeted=False, device=None):
       
        super().__init__(model, epsilon, device)
        self.alpha = alpha
        self.iterations = iterations
        self.random_start = random_start
        self.targeted = targeted
        
        # alpha should be <= epsilon/iterations for stability
        if alpha > epsilon / iterations:
            print(f"Warning: alpha={alpha} might be too large for ε={epsilon}, iterations={iterations}")
    
    def attack(self, images, labels, target_labels=None):
        """
        Generate PGD adversarial examples
        
        Args:
            images: clean images
            labels: true labels
            target_labels: target labels for targeted attack
        
        Returns:
            adversarial_images
        """
        images, labels = self._check_input(images, labels)
        
        
        adversarial_images = images.clone().detach()
        
       
        if self.random_start:
            # Start from random point within epsilon ball
            random_noise = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            adversarial_images = torch.clamp(images + random_noise, 0, 1)
        
        # PGD iterations
        for i in range(self.iterations):
           
            adversarial_images.requires_grad = True
            was_training = self.model.training
            self.model.train()
            outputs = self.model(adversarial_images)
            
            if self.targeted:
                # Targeted: minimize loss for target class
                if target_labels is None:
                    raise ValueError("target_labels required for targeted attack")
                loss = F.cross_entropy(outputs, target_labels)
            else:
                # Untargeted: maximize loss for true class
                loss = -F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            gradient = adversarial_images.grad.data
            
            with torch.no_grad():
                if self.targeted:
                    # Move TOWARD target class
                    adversarial_images = adversarial_images - self.alpha * torch.sign(gradient)
                else:
                    # Move AWAY from true class
                    adversarial_images = adversarial_images + self.alpha * torch.sign(gradient)
                
                
                delta = adversarial_images - images
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adversarial_images = images + delta
                
                
                adversarial_images = torch.clamp(adversarial_images,0,1)
            
            adversarial_images = adversarial_images.detach()
        
        return adversarial_images
    
    def __str__(self):
        return f"PGD(ε={self.epsilon}, α={self.alpha}, iter={self.iterations})"