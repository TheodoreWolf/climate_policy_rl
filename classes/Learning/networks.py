"""
@Theodore Wolf
A few simple networks that can be used by different types of agents
"""

import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Outputs action preferences, to be used by Actor-critics"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNet, self).__init__()
        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.a = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        l = self.layer(obs)
        out = self.a(l)

        return out


class Net(nn.Module):
    """Outputs Q-values for each action"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Net, self).__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.q = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        l = self.layer(obs)
        q_values = self.q(l)

        return q_values


class DuellingNet(nn.Module):
    """Outputs duelling Q-values for each action"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuellingNet, self).__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   #nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
                                   )

        self.a = nn.Linear(hidden_dim, action_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        l = self.layer(obs)
        advantages = self.a(l)
        value = self.v(l)
        a_values = value + (advantages - advantages.mean())

        return a_values


class DualACNET(nn.Module):
    """Outputs both the value for the state and the action preferences"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DualACNET, self).__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.dist = nn.Linear(hidden_dim, action_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        l = self.layer(obs)
        policy_dist= F.softmax(self.dist(l), dim=-1)
        value = self.v(l)

        return value, policy_dist
