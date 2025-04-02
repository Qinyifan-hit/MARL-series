import torch.nn as nn
import numpy as np
import torch


def mapping(a, up, low):
    a = (up + low) / 2 + a * (up - low) / 2
    return a


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class A_net(nn.Module):
    def __init__(self, state_n, action_n, a_up, a_low, width):
        super(A_net, self).__init__()
        self.a_up = a_up
        self.a_low = a_low
        self.Act = nn.Sequential(
            layer_init(nn.Linear(state_n, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, action_n)),
            nn.Tanh()
        )

    def forward(self, s):
        a = self.Act(s)
        a = mapping(a, self.a_up, self.a_low)
        return a


class C_net(nn.Module):
    def __init__(self, state_n, action_n, width):
        super(C_net, self).__init__()
        self.Q1_net = nn.Sequential(
            layer_init(nn.Linear(state_n + action_n, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, 1)),
            nn.Identity()
        )

        self.Q2_net = nn.Sequential(
            layer_init(nn.Linear(state_n + action_n, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, 1)),
            nn.Identity()
        )

    def forward(self, s, a):
        Input = torch.cat([s, a], dim=-1)
        Q1 = self.Q1_net(Input)
        Q2 = self.Q2_net(Input)
        return Q1, Q2











