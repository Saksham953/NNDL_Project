import torch
import torch.nn.functional as F

def reconstruction_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

def discriminator_loss(real_pred, fake_pred):
    # LSGAN (Least Squares GAN) Loss
    real_loss = torch.mean((real_pred - 1) ** 2)
    fake_loss = torch.mean(fake_pred ** 2)
    return (real_loss + fake_loss) * 0.5

def generator_adversarial_loss(fake_pred):
    return torch.mean((fake_pred - 1) ** 2)