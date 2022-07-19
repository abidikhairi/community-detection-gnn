import torch as th
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, nfeats, ncomms) -> None:
        super().__init__()

        self.linear = nn.Linear(nfeats, ncomms, bias=False)

    def forward(self, x):
        return self.linear(x)
