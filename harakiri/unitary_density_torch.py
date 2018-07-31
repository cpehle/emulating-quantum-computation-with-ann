import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from . import unitary_density
from . import quantum as qm

num_samples = 1000
data = unitary_density.generate_data(qm.hadamard, n=np.shape(qm.hadamard)[0], num_samples=num_samples)


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

net = LinearNet()
print(net)