import torch
import torch.nn.functional as F
import copy
import numpy as np
from Net_con import A_net, C_net
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DDPG_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.a_dim = self.action_dim[self.id]
        self.s_dim = self.state_dim[self.id]
        self.a_up = self.action_up[self.id]
        self.a_low = self.action_low[self.id]

        self.Actor = A_net(self.s_dim, self.a_dim, self.a_up, self.a_low, self.hidden_sizes).to(device)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.a_lr)
        self.A_target = copy.deepcopy(self.Actor)

        self.Critic = C_net(sum(self.state_dim), sum(self.action_dim), self.hidden_sizes).to(device)
        self.C_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.c_lr)
        self.C_target = copy.deepcopy(self.Critic)

        self.noise = self.noise_scale

    def action_select(self, s, iseval):
        with torch.no_grad():
            s = torch.FloatTensor(s).view(1, -1).to(device)
            a = self.Actor(s).cpu().squeeze(0).numpy()
            if iseval:
                return a
            else:
                noise = np.random.normal(0, self.noise * self.a_up, self.a_dim)
                a = np.clip(a + noise, self.a_low, self.a_up)
                return np.array(a, dtype=np.float32)

    def train(self, s, a, r, s_, done, a_, agent_id):
        s_full = torch.cat(s, dim=-1)
        a_full = torch.cat(a, dim=-1)
        s_full_ = torch.cat(s_, dim=-1)
        a_full_ = torch.cat(a_, dim=-1)

        with torch.no_grad():
            Q_ = self.C_target(s_full_, a_full_)
            Q_target = r[agent_id] + (~done[agent_id]) * self.gamma * Q_

        Q = self.Critic(s_full, a_full)
        C_loss = F.mse_loss(Q, Q_target)
        self.C_optimizer.zero_grad()
        C_loss.backward()
        self.C_optimizer.step()

        a_pred = []
        for a_id in range(self.Num_agent):
            if a_id == agent_id:
                a_pred.append(self.Actor(s[a_id]))
            else:
                a_pred.append(a[a_id].detach())
        a_pred = torch.cat(a_pred, dim=-1)
        A_loss = torch.mean(-self.Critic(s_full, a_pred))
        self.A_optimizer.zero_grad()
        A_loss.backward()
        self.A_optimizer.step()

        for p, p_tar in zip(self.Actor.parameters(), self.A_target.parameters()):
            p_tar.data.copy_(self.tau * p + (1 - self.tau) * p_tar)

        for p, p_tar in zip(self.Critic.parameters(), self.C_target.parameters()):
            p_tar.data.copy_(self.tau * p + (1 - self.tau) * p_tar)

    def load(self, BName, Index, agent_id):
        self.Actor.load_state_dict(
            torch.load("./model/{}_Actor_{}_{}.pth".format(BName, agent_id, Index), map_location=device))
        self.Critic.load_state_dict(
            torch.load("./model/{}_Critic_{}_{}.pth".format(BName, agent_id, Index), map_location=device))
        self.C_target = copy.deepcopy(self.Critic)
        self.A_target = copy.deepcopy(self.Actor)

    def save(self, BName, Index, agent_id):
        torch.save(self.Actor.state_dict(), "./model/{}_Actor_{}_{}.pth".format(BName, agent_id, Index))
        torch.save(self.Critic.state_dict(), "./model/{}_Critic_{}_{}.pth".format(BName, agent_id, Index))
