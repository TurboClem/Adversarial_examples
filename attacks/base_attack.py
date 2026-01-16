import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttack(nn.Module):
    """
    Abstract base class for adversarial attacks

    """

    def __init__(self, model, epsilon=0.03, device=None):
        """
        Args:
            model: PyTorch model to attack
            epsilon: maximum perturbation (L∞ norm)
            device: device to run on (cuda/cpu)
        """
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

        self.model.eval()  # Important: attack in eval mode

    def attack(self, images, labels, **kwargs):
        """
        Generate adversarial examples

        Args:
            images: clean images [B, C, H, W]
            labels: true labels [B]

        Returns:
            adversarial_images: perturbed images
        """
        raise NotImplementedError

    def _clip_values(self, perturbed, original):
        """
        Clip perturbed images to valid range

        Args:
            perturbed: perturbed images
            original: original images (for reference)

        Returns:
            clipped images within epsilon ball
        """
        # Clip perturbation to epsilon ball (L∞ norm)
        delta = perturbed - original
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)

        # Add clipped perturbation to original
        perturbed = original + delta

        # Additional clipping to [0, 1] after denormalization check
        # (Images are normalized with ImageNet stats)
        return perturbed

    def _check_input(self, images, labels):
        """Validate input tensors"""
        assert images.dim() == 4, "Images must be 4D tensor [B, C, H, W]"
        assert labels.dim() == 1, "Labels must be 1D tensor [B]"
        assert images.size(0) == labels.size(0), "Batch size mismatch"

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        return images, labels

    def compute_success_rate(self, clean_images, adv_images, labels):
        """
        Compute attack success rate

        Returns:
            clean_acc: accuracy on clean images
            adv_acc: accuracy on adversarial images
            success_rate: percentage of successful attacks
        """
        with torch.no_grad():

            # Clean accuracy
            clean_outputs = self.model(clean_images)
            clean_preds = torch.argmax(clean_outputs, dim=1)
            clean_acc = (clean_preds == labels).float().mean()

            # Adversarial accuracy
            adv_outputs = self.model(adv_images)
            adv_preds = torch.argmax(adv_outputs, dim=1)
            adv_acc = (adv_preds == labels).float().mean()

            # Attack success rate
            success_rate = 1.0 - adv_acc

        return clean_acc.item(), adv_acc.item(), success_rate.item()
