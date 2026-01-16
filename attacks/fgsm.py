"""
FGSM Attack: Fast Gradient Sign Method
Paper: "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2015)

Key idea: One-step attack in the direction of gradient sign
x_adv = x + ε * sign(∇_x L(θ, x, y))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attack import BaseAttack


class FGSM(BaseAttack):
    """
    Fast Gradient Sign Method attack

    Simple one-step attack:
    1. Compute gradient of loss w.r.t input
    2. Take sign of gradient
    3. Perturb image: x_adv = x + ε * sign(gradient)
    """

    def __init__(self, model, epsilon=0.03, targeted=False, device=None):
        """
        Args:
            model: model to attack
            epsilon: perturbation magnitude (L∞ norm)
            targeted: if True, perform targeted attack
            device: device to run on
        """
        super().__init__(model, epsilon, device)
        self.targeted = targeted

    def attack(self, images, labels, target_labels=None):
        """
        Generate FGSM adversarial examples

        Args:
            images: clean images
            labels: true labels for untargeted attack
            target_labels: target labels for targeted attack

        Returns:
            adversarial_images
        """
        images, labels = self._check_input(images, labels)

        # Enable gradient computation for input images
        images.requires_grad = True

        # Forward pass
        outputs = self.model(images)

        # Calculate loss
        if self.targeted:
            # Targeted attack: minimize loss for target class
            if target_labels is None:
                raise ValueError("target_labels required for targeted attack")
            loss = F.cross_entropy(outputs, target_labels)
        else:
            # Untargeted attack: maximize loss for true class
            loss = -F.cross_entropy(outputs, labels)

        # Zero gradients, then backward
        self.model.zero_grad()
        loss.backward()

        # Get gradient with respect to input
        gradient = images.grad.data

        # FGSM perturbation: ε * sign(gradient)
        perturbation = self.epsilon * torch.sign(gradient)

        # Apply perturbation
        if self.targeted:
            # Move TOWARD target class
            adversarial_images = images - perturbation
        else:
            # Move AWAY from true class
            adversarial_images = images + perturbation

        # Clip to valid range
        adversarial_images = self._clip_values(adversarial_images, images.detach())

        return adversarial_images.detach()

    def __str__(self):
        return f"FGSM(ε={self.epsilon}, targeted={self.targeted})"
