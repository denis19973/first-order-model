import torch.nn as nn
import torchvision
import numpy as np


imgnet_mean = np.array([0.485, 0.456, 0.406])
imgnet_std  = np.array([0.229, 0.224, 0.225])


IMGNET_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(imgnet_mean, imgnet_std)
])


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)
