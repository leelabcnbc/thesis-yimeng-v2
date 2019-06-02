from torch import nn
import numpy as np


def count_params(model: nn.Module):
    count = 0
    for y in model.parameters():
        count += np.product(y.size())
    return count
