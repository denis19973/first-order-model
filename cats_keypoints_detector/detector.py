import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from cats_keypoints_detector.utils import Reshape


class Detector(torch.nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.base_model = torchvision.models.mobilenet_v2(pretrained=True)
        classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=18),
            Reshape(9, 2)
        )
        self.base_model.classifier = classifier

    def forward(self, x):
        x = self.base_model(x)
        return x

    def load_my_state_dict(self, state_dict):
        self.base_model.load_state_dict(state_dict)
