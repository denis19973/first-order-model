import torch
from cats_keypoints_detector.utils import Reshape


class Mapper(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Mapper, self).__init__()
        self.seq = torch.nn.Sequential(
        torch.nn.Linear(inputSize, (inputSize + outputSize) // 2),
        #torch.nn.Dropout(p=0.1),
        torch.nn.ReLU(),
        torch.nn.Linear((inputSize + outputSize) // 2, (inputSize + outputSize) // 2),
        #torch.nn.Dropout(p=0.1),
        torch.nn.ReLU(),
        torch.nn.Linear((inputSize + outputSize) // 2, (inputSize + outputSize) // 2),
        # torch.nn.Dropout(p=0.05),
        #torch.nn.ReLU(),
        #torch.nn.Linear((inputSize + outputSize) // 2, (inputSize + outputSize) // 2),
        #torch.nn.Dropout(p=0.2),
        torch.nn.ReLU(),
        torch.nn.Linear((inputSize + outputSize) // 2, outputSize),
        Reshape(outputSize // 2, 2)
        )

    def forward(self, x):
        x = self.seq(x)
        return x
