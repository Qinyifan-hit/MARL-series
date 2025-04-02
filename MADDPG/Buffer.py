import torch
import numpy as np
import copy

dic = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Replay_Buffer(object):
    def __init__(self, action_n, state_n, max_size, agents, Num_agent):
        self.pr = 0
        self.size = 0
        self.action_n = action_n
        self.state_n = state_n
        self.max_size = int(max_size)
        self.agents = agents
        self.Num_agent = Num_agent

        self.s = []
        self.a = []
        self.s_ = []
        self.r = []
        self.done = []
        self.dw = []

        for a_id in range(Num_agent):
            self.s.append(np.zeros((self.max_size, self.state_n[a_id]), dtype=np.float32))
            self.a.append(np.zeros((self.max_size, self.action_n[a_id]), dtype=np.float32))
            self.r.append(np.zeros((self.max_size, 1), dtype=np.float32))
            self.done.append(np.zeros((self.max_size, 1), dtype=np.bool_))

        self.s_ = copy.deepcopy(self.s)
        self.dw = copy.deepcopy(self.done)


    def add(self, s, a, r, s_, done, dw):
        for a_id in range(self.Num_agent):
            self.s[a_id][self.pr] = s[a_id]
            self.a[a_id][self.pr] = a[a_id]
            self.r[a_id][self.pr] = r[a_id]
            self.s_[a_id][self.pr] = s_[a_id]
            self.done[a_id][self.pr] = done[a_id]
            self.dw[a_id][self.pr] = dw[a_id]

        self.pr = (self.pr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def B_sample(self, b_size):
        Ind = np.random.choice(self.size, b_size, replace=True)
        s = []
        a = []
        r = []
        s_ = []
        dw = []
        for agent_id in range(self.Num_agent):
            s.append(torch.FloatTensor(self.s[agent_id][Ind]).to(dic))
            a.append(torch.FloatTensor(self.a[agent_id][Ind]).to(dic))
            r.append(torch.FloatTensor(self.r[agent_id][Ind]).to(dic))
            s_.append(torch.FloatTensor(self.s_[agent_id][Ind]).to(dic))
            dw.append(torch.BoolTensor(self.dw[agent_id][Ind]).to(dic))

        return s, a, r, s_, dw
