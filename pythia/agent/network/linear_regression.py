import torch
import numpy as np
from torch.autograd import Variable


class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x: torch.Tensor):
        out = self.linear(x)
        return out