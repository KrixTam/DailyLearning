import torch
from torch import nn


class RBFNetwork(nn.Module):
    def __init__(self, clusters):
        super(RBFNetwork, self).__init__()
        # remember how many centers we have
        self.N = clusters.shape[0]
        # our mean and sigmas for the RBF layer
        self.sigs = nn.Parameter(torch.ones(self.N, dtype=torch.float64) * 5, requires_grad=False)  # our sigmas
        self.mus = nn.Parameter(torch.from_numpy(clusters), requires_grad=False)  # our means

        self.linear = nn.Linear(self.N, 10, dtype=torch.float64)

    def forward(self, x):
        distances = torch.sqrt(((x.unsqueeze(1) - self.mus) ** 2).sum(dim=2))
        # Calculate the Gaussian activations
        res = torch.exp((-0.5) * (distances ** 2) / self.sigs ** 2)
        # Set any NaN values to 0 (in case self.sigs is zero)
        res[res != res] = 0.0

        out = self.linear(res)
        return out