from torch import nn


class LambdaSingle(nn.Module):
    """this module does nothing. the main purpose is to facilitate feature extraction"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
