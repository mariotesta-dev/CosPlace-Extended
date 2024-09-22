import torch
import torch.nn as nn
import torchvision.transforms as transforms

'''
This code has been extracted from adageo-WACV2021 GitHub Repository:
https://github.com/valeriopaolicelli/adageo-WACV2021
'''

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # during the forward pass, GRL acts as an identity transform (it just copy and paste data without changing anything)
        return x.clone()
    @staticmethod
    def backward(ctx, grads):
        # during the backward pass, GRL gets gradient of the subsequent level, 
        # multiplies it by -lambda (trade-off between Ly and Ld) and passes it to the precedent level
        dx = -grads.new_tensor(1) * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.shape[0], -1)
        return GradientReversalFunction.apply(x)


def get_discriminator(input_dim, num_classes=2):
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(input_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes)
    )
    return discriminator

grl_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

