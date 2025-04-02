import torch.nn as nn
import numpy as np
import torch

def mapping(a, up, low):
    a = (up + low) / 2 + a * (up - low) / 2
    return a

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class A_net(nn.Module):
    def __init__(self, state_n, action_n, up, low, width):
        super(A_net, self).__init__()
        self.A_net = nn.Sequential(
            layer_init(nn.Linear(state_n, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, action_n)),
            nn.Tanh()
        )
        self.a_up = up
        self.a_low = low

    def forward(self, s):
        a = self.A_net(s)
        a = mapping(a, self.a_up, self.a_low)
        return a


class C_net(nn.Module):
    def __init__(self, state_n, action_n, width):
        super(C_net, self).__init__()
        self.Q_net = nn.Sequential(
            layer_init(nn.Linear(state_n + action_n, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, 1)),
            nn.Identity()
        )

    def forward(self, s, a):
        Input = torch.cat([s, a], -1)
        return self.Q_net(Input)
