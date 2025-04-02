import torch
import torch.nn.functional as F
import copy
import numpy as np
from Net_con import A_net, C_net

dic = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TD3_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.state_n = self.state_dim[self.agent_id]
        self.action_n = self.action_dim[self.agent_id]
        self.a_up = self.action_up[self.agent_id]
        self.a_low = self.action_low[self.agent_id]

        self.Actor = A_net(self.state_n, self.action_n, self.a_up, self.a_low, self.hidden_sizes).to(dic)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.a_lr)
        self.A_target = copy.deepcopy(self.Actor)

        self.Q_mix = C_net(sum(self.state_dim), sum(self.action_dim), self.hidden_sizes).to(dic)
        self.Q_optimizer = torch.optim.Adam(self.Q_mix.parameters(), lr=self.c_lr)
        self.Q_target = copy.deepcopy(self.Q_mix)

        self.noise = self.noise_scale

        self.policy_noise = 0.2 * self.a_up
        self.noise_clip = 0.5 * self.a_up


    def action_select(self, s, iseval):
        with torch.no_grad():
            s = torch.FloatTensor(s).view(1, -1).to(dic)
            a = self.Actor(s).cpu().numpy().squeeze(0)
            if iseval:
                return np.array(a, dtype=np.float32)
            else:
                noise = np.random.normal(0, self.noise * (self.a_up - self.a_low) / 2, self.action_n)
                a = np.clip(a + noise, self.a_low, self.a_up)
                return np.array(a, dtype=np.float32)

    def train(self, s, a, r, s_, dw, a_, agent_id):
        # noise:
        Num_agent = len(a_)
        noise_a_ = [torch.clip(torch.randn_like(a_[j]) * self.policy_noise, -self.noise_clip, self.noise_clip) for j in range(Num_agent)]
        a_n_ = [torch.clip(a_[j] + noise_a_[j], self.a_low, self.a_up) for j in range(Num_agent)]
        s_full = torch.cat(s, dim=-1)
        a_full = torch.cat(a, dim=-1)
        s_full_ = torch.cat(s_, dim=-1)
        a_full_n_ = torch.cat(a_n_, dim=-1)
        with torch.no_grad():
            Q1_, Q2_ = self.Q_target(s_full_, a_full_n_)
            Q_ = torch.min(Q1_, Q2_)
            Q_target = r[agent_id] + self.gamma * Q_ * (~dw[agent_id])

        Q1, Q2 = self.Q_mix(s_full, a_full)
        Q_loss = F.mse_loss(Q1, Q_target) + F.mse_loss(Q2, Q_target)
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        a_pred = []
        for a_id in range(Num_agent):
            if a_id == agent_id:
                a_pred.append(self.Actor(s[a_id]))
            else:
                a_pred.append(a[a_id].detach())
        a_pred = torch.cat(a_pred, dim=-1)
        Q1_pred, _ = self.Q_mix(s_full, a_pred)
        A_loss = torch.mean(-Q1_pred)
        self.A_optimizer.zero_grad()
        A_loss.backward()
        self.A_optimizer.step()

        for p, p_tar in zip(self.Actor.parameters(), self.A_target.parameters()):
            p_tar.data.copy_(self.tau * p + (1 - self.tau) * p_tar)

        for p, p_tar in zip(self.Q_mix.parameters(), self.Q_target.parameters()):
            p_tar.data.copy_(self.tau * p + (1 - self.tau) * p_tar)


    def load(self, BName, agent_id, Index):
        self.Actor.load_state_dict(torch.load("./model/{}_Actor_{}_{}.pth".format(BName, agent_id, Index), map_location=dic))
        self.Q_mix.load_state_dict(torch.load("./model/{}_Critic_{}_{}.pth".format(BName, agent_id, Index), map_location=dic))
        self.A_target = copy.deepcopy(self.Actor)
        self.Q_target = copy.deepcopy(self.Q_mix)

    def save(self, BName, Index, agent_id):
        torch.save(self.Actor.state_dict(), "./model/{}_Actor_{}_{}.pth".format(BName, agent_id, Index))
        torch.save(self.Q_mix.state_dict(), "./model/{}_Critic_{}_{}.pth".format(BName, agent_id, Index))

