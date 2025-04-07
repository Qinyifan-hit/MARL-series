import numpy as np
import torch
import copy

dic = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class traj_record(object):
    def __init__(self, batch_size, obs_dim, act_dim, agents, max_steps):
        self.batch_size = batch_size
        self.obs_n = obs_dim[0]
        self.act_n = act_dim[0]
        self.agents = agents
        self.N_agents = len(agents)
        self.max_steps = max_steps

        self.obs = np.zeros([self.batch_size, self.max_steps, self.N_agents, self.obs_n], dtype=np.float32)
        self.act = np.zeros([self.batch_size, self.max_steps, self.N_agents, self.act_n], dtype=np.float32)
        self.log_probs = copy.deepcopy(self.act)
        self.reward = np.zeros([self.batch_size, self.max_steps, self.N_agents, 1], dtype=np.float32)
        self.value = np.zeros([self.batch_size, self.max_steps, self.N_agents, 1], dtype=np.float32)
        self.value_ = copy.deepcopy(self.value)
        self.obs_ = copy.deepcopy(self.obs)
        self.state = np.zeros([self.batch_size, self.max_steps, int(sum(obs_dim))], dtype=np.float32)
        self.state_ = copy.deepcopy(self.state)
        self.done = np.zeros([self.batch_size, self.max_steps, self.N_agents, 1], dtype=np.bool_)

        self.size = 0

    def update(self, obs, a, r, obs_, s, s_, done, log_probs, value, value_, Ind):
        for a_id in range(self.N_agents):
            self.obs[self.size][Ind][a_id] = obs[a_id]
            self.act[self.size][Ind][a_id] = a[a_id]
            self.reward[self.size][Ind][a_id] = r[a_id]
            self.obs_[self.size][Ind][a_id] = obs_[a_id]
            self.done[self.size][Ind][a_id] = done[a_id]
            self.log_probs[self.size][Ind][a_id] = log_probs[a_id]
            self.value[self.size][Ind][a_id] = value[a_id]
            self.value_[self.size][Ind][a_id] = value_[a_id]

        self.state[self.size][Ind] = s
        self.state_[self.size][Ind] = s_

        if Ind == int(self.max_steps - 1):
            self.size += 1

    def read(self):
        return(
            torch.FloatTensor(self.obs).to(dic),
            torch.FloatTensor(self.act).to(dic),
            torch.FloatTensor(self.log_probs).to(dic),
            torch.FloatTensor(self.reward).to(dic),
            torch.FloatTensor(self.obs_).to(dic),
            torch.FloatTensor(self.state).to(dic),
            torch.FloatTensor(self.state_).to(dic),
            torch.FloatTensor(self.value).to(dic),
            torch.FloatTensor(self.value_).to(dic),
            torch.BoolTensor(self.done).to(dic)
        )





