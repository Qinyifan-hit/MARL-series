import torch
import torch.nn.functional as F
import numpy as np
import copy
import math
from Net_con import A_net_beta, C_net

dic = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MAPPO_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.act_input = self.obs_dim[0]
        self.cri_input = self.state_dim
        self.action_n = self.action_dim[0]

        self.Actor = A_net_beta(self.act_input, self.action_n, self.mlp_hidden_dim).to(dic)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.lr)

        self.Critic = C_net(self.cri_input, self.mlp_hidden_dim).to(dic)
        self.C_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.lr)

    def action_select(self, s, iseval):
        with torch.no_grad():
            s = torch.FloatTensor(s).view(1, -1).to(dic)
            if iseval:
                a, _, _ = self.Actor(s, True)
                return a.cpu().numpy().squeeze(0)
            else:
                a, log_prob, _ = self.Actor(s, False)
                return a.cpu().numpy().squeeze(0), log_prob.cpu().numpy().squeeze(0)


    def get_value(self, s_full):
        with torch.no_grad():
            s_full = torch.FloatTensor(s_full).view(1, -1).to(dic)
            value = []
            for a_id in range(self.Num_agent):
                V = self.Critic(s_full)
                value.append(V.cpu().numpy().squeeze(0))
            return value

    def train(self, Traj):
        obs, act, log_prob_old, r, obs_, s, s_, V, V_, done = Traj.read()
        e_turns = int(self.batch_size / self.mini_batch_size)
        with torch.no_grad():
            deltas = r + self.gamma * V_ * (~done) - V
            A = 0
            Adv_list = []
            for t in range(self.max_steps - 1, -1, -1):
                A = deltas[:, t] + self.gamma * self.lamda * (~done[:, t]) * A
                Adv_list.insert(0, A)
            Adv = torch.stack(Adv_list, dim=1)
            V_target = Adv + V
            Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-5)

        for _ in range(self.K_epochs):
            Ind = np.arange(self.batch_size)
            np.random.shuffle(Ind)
            # C_loss
            s, V_target, V = s[Ind].clone(), V_target[Ind].clone(), V[Ind].clone()
            # A_loss
            log_prob_old, obs, act, Adv = log_prob_old[Ind].clone(), obs[Ind].clone(), act[Ind].clone(), Adv[Ind].clone()
            for j in range(e_turns):
                Ind_b = slice(j * self.mini_batch_size, min(self.batch_size, (j + 1) * self.mini_batch_size))
                _, _, Distri_new = self.Actor(obs[Ind_b], False)
                log_prob_new = Distri_new.log_prob(act[Ind_b])
                Entropy = Distri_new.entropy().sum(-1, keepdim=True)

                r_t = (log_prob_new.sum(-1, keepdim=True) - log_prob_old[Ind_b].sum(-1, keepdim=True)).exp()
                A_L1 = r_t * Adv[Ind_b]
                A_L2 = torch.clip(r_t, 1 - self.clip_rate, 1 + self.clip_rate) * Adv[Ind_b]
                A_loss = (-torch.min(A_L2, A_L1) - self.entropy_coef * Entropy).mean()
                self.A_optimizer.zero_grad()
                A_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), 10.0)
                self.A_optimizer.step()

                V_b = self.Critic(s[Ind_b]).unsqueeze(-1)
                C_L1 = (V_b - V_target[Ind_b]).pow(2)
                C_L2 = (torch.clip(V_b, V[Ind_b] - self.clip_rate, V[Ind_b] + self.clip_rate) - V_target[Ind_b]).pow(2)
                C_loss = (torch.max(C_L1, C_L2)).mean()
                self.C_optimizer.zero_grad()
                C_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), 10.0)
                self.C_optimizer.step()

    def load(self, BName, Ind):
        self.Actor.load_state_dict(torch.load("./model/{}_Actor_{}.pth".format(BName, Ind), map_location=dic))
        self.Critic.load_state_dict(torch.load("./model/{}_Critic_{}.pth".format(BName, Ind), map_location=dic))


    def save(self, BName, Ind):
        torch.save(self.Actor.state_dict(), "./model/{}_Actor_{}.pth".format(BName, Ind))
        torch.save(self.Critic.state_dict(), "./model/{}_Critic_{}.pth".format(BName, Ind))














































