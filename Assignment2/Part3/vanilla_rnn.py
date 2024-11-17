from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.hidden_dim = hidden_dim

        self.Wxh = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_dim))

        self.Why = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)


    def forward(self, x):
        # Implementation here ...
        bach_size = x.size(0)
        h_t = torch.zeros(bach_size, self.hidden_dim, device=x.devcie)

        for t in range(x.size(1)):
            x_t = x[:, t, :]
            h_t = torch.tanh(x_t @ self.Wxh + h_t @ self.Whh + self.bh)

        y = h_t @ self.Why + self.by
        return y
        
    # add more methods here if needed
