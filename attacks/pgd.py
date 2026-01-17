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

    def __init__(
        self,
        model,
        epsilon=0.03,
        alpha=0.01,
        iterations=10,
        random_start=True,
        seed=42,
        targeted=False,
        device=None,
    ):

        super().__init__(model, epsilon, device)
        self.epsilon = epsilon.to(device) if torch.is_tensor(epsilon) else epsilon
        self.alpha = alpha.to(device) if torch.is_tensor(alpha) else alpha
        self.iterations = iterations
        self.random_start = random_start
        self.rng = torch.Generator(device=device)
        if seed:
            self.rng.manual_seed(seed)
        self.targeted = targeted

        from config import MEAN, STD
        
        # Convert to tensors on the right device
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.clip_min = (0 - self.mean) / self.std
        self.clip_max = (1 - self.mean) / self.std

        # alpha should be <= epsilon/iterations for stability
        if torch.any(alpha > epsilon / iterations):
            print(
                f"Warning: alpha={alpha} might be too large for ε={epsilon}, iterations={iterations}"
            )

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
            #random_noise = (
            #    torch.rand(images.shape, device=images.device, generator=self.rng) * 2
            #    - 1
            #) * 1e-5 * self.epsilon
            # random_noise = torch.empty_like(images, generator=self.rng).uniform_(-self.epsilon, self.epsilon)
            adversarial_images = torch.clamp(
                 images, images.min(), images.max()  # + random_noise, images.min(), images.max()
            )  # min, max = 0, 1?

        # PGD iterations
        for i in range(self.iterations):

            adversarial_images.requires_grad = True
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
            if adversarial_images.grad is not None:
                adversarial_images.grad.zero_()
            loss.backward()
            gradient = adversarial_images.grad.data

            #gradient = torch.autograd.grad(loss, adversarial_images)[0]
            g_std = torch.std(gradient, dim=[1, 2, 3], keepdim=True)
            g_mean = torch.mean(gradient, dim=[1, 2, 3], keepdim=True)
            gradient = gradient - g_mean
            gradient = gradient / (g_std + 1e-8)

            with torch.no_grad():
                if self.targeted:
                    # Move TOWARD target class
                    #adversarial_images = adversarial_images - self.alpha * torch.sign(
                    #    gradient
                    #)
                    adversarial_images = adversarial_images - self.alpha * gradient
                else:
                    # Move AWAY from true class
                    adversarial_images = adversarial_images + self.alpha * torch.sign(
                        gradient
                    )
                    #adversarial_images = adversarial_images + self.alpha * gradient

                delta = adversarial_images - images
                # delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                #delta = torch.max(torch.min(delta, self.epsilon), -self.epsilon)
                delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)

                adversarial_images = images + delta

                #adversarial_images = torch.clamp(
                #    adversarial_images, images.min(), images.max()
                #)

            adversarial_images = adversarial_images.detach()

        return adversarial_images

    def __str__(self):
        return f"PGD(ε={self.epsilon}, α={self.alpha}, iter={self.iterations})"


class PGD_robust_analysis(BaseAttack):
    def __init__(
        self,
        model,
        epsilon=0.03,
        alpha=0.01,
        iterations=10,
        random_start=True,
        seed=42,
        targeted=False,
        device=None,
    ):
        super().__init__(model, epsilon, device)
        
        # Handle epsilon - can be scalar or per-channel tensor
        if torch.is_tensor(epsilon):
            self.epsilon = epsilon.to(device)
        else:
            # Convert scalar to tensor with 3 channels
            self.epsilon = torch.tensor([epsilon, epsilon, epsilon], device=device)
        
        # Handle alpha - can be scalar or per-channel tensor
        if torch.is_tensor(alpha):
            self.alpha = alpha.to(device)
        else:
            # Convert scalar to tensor with 3 channels
            self.alpha = torch.tensor([alpha, alpha, alpha], device=device)
            
        self.iterations = iterations
        self.random_start = random_start
        self.targeted = targeted
        
        from config import MEAN, STD
        
        # Store normalization parameters
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        
        # Reshape epsilon and alpha for broadcasting
        self.epsilon = self.epsilon.view(1, 3, 1, 1)
        self.alpha = self.alpha.view(1, 3, 1, 1)
        
        # Scale epsilon and alpha by std for normalized space
        self.epsilon_scaled = self.epsilon / self.std
        self.alpha_scaled = self.alpha / self.std
        
        # Check alpha <= epsilon/iterations
        epsilon_vals = self.epsilon.view(-1)
        alpha_vals = self.alpha.view(-1)
        if torch.any(alpha_vals > epsilon_vals / iterations):
            print(f"Warning: alpha={alpha_vals.tolist()} might be too large for ε={epsilon_vals.tolist()}, iterations={iterations}")

    def attack(self, images, labels, target_labels=None):
        """
        Generate PGD adversarial examples
        """
        images, labels = self._check_input(images, labels)
        
        # Clone images
        adv_images = images.clone().detach()
        
        # Random start within epsilon ball (in normalized space)
        if self.random_start:
            # Generate random noise with per-channel bounds
            noise = torch.empty_like(adv_images)
            for c in range(3):
                noise[:, c:c+1, :, :].uniform_(
                    -self.epsilon_scaled[0, c, 0, 0].item(),
                    self.epsilon_scaled[0, c, 0, 0].item()
                )
            
            adv_images = adv_images + noise
            
            # Clip to valid normalized range
            self._clip(adv_images)
        
        # PGD iterations
        for i in range(self.iterations):
            adv_images.requires_grad = True
            
            # Forward pass
            outputs = self.model(adv_images)
            
            # Calculate loss
            if self.targeted:
                if target_labels is None:
                    raise ValueError("target_labels required for targeted attack")
                loss = F.cross_entropy(outputs, target_labels)
            else:
                loss = -F.cross_entropy(outputs, labels)
            
            # Compute gradient
            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            loss.backward()
            
            # Get gradient
            gradient = adv_images.grad.data
            
            # Use sign gradient for stronger attack
            gradient = torch.sign(gradient)
            
            # Update step
            with torch.no_grad():
                if self.targeted:
                    # Move TOWARD target class
                    adv_images = adv_images - self.alpha_scaled * gradient
                else:
                    # Move AWAY from true class
                    adv_images = adv_images + self.alpha_scaled * gradient
                
                # Project back to epsilon ball (in normalized space)
                delta = adv_images - images
                delta = torch.clamp(delta, 
                                  min=-self.epsilon_scaled, 
                                  max=self.epsilon_scaled)
                
                adv_images = images + delta
                
                # Clip to valid normalized range
                self._clip(adv_images)
            
            adv_images = adv_images.detach()
        
        return adv_images
    
    def _clip(self, images):
        """Clip images to valid normalized range"""
        min_val = (0 - self.mean) / self.std
        max_val = (1 - self.mean) / self.std
        images.data = torch.clamp(images, min_val, max_val)
    
    def __str__(self):
        epsilon_vals = self.epsilon.view(-1).tolist()
        alpha_vals = self.alpha.view(-1).tolist()
        if len(set(epsilon_vals)) == 1 and len(set(alpha_vals)) == 1:
            return f"PGD(ε={epsilon_vals[0]}, α={alpha_vals[0]}, iter={self.iterations})"
        else:
            return f"PGD(ε={epsilon_vals}, α={alpha_vals}, iter={self.iterations})"

