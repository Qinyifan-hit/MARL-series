import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class A_net_beta(nn.Module):
    def __init__(self, input_dim, action_n, width):
        super(A_net_beta, self).__init__()
        self.A_net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh()
        )
        self.alpha = nn.Linear(width, action_n)
        self.beta = nn.Linear(width, action_n)

    def forward(self, s, isdeter):
        h = self.A_net(s)
        alpha = F.softplus(self.alpha(h)) + 1.0
        beta = F.softplus(self.beta(h)) + 1.0

        if isdeter:
            action = alpha / (alpha + beta)
            log_probs = None
            distri = None
        else:
            distri = Beta(alpha, beta)
            action = distri.sample()
            log_probs = distri.log_prob(action)

        return action, log_probs, distri


class C_net(nn.Module):
    def __init__(self, Input, net_width):
        super(C_net, self).__init__()
        self.C_net = nn.Sequential(
            nn.Linear(Input, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1),
            nn.Identity()
        )

    def forward(self, s):
        V = self.C_net(s)
        return V
