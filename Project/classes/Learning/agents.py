import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.q = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        l = self.layer(obs)
        q_values = self.q(l)

        return q_values


class DuellingNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.a = nn.Linear(hidden_dim, action_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        l = self.layer(obs)
        advantages = self.a(l)
        value = self.v(l)

        q_values = value + (advantages - advantages.mean())
        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class agent:
    def __init__(self, state_dim, action_dim, gamma=0.96, alpha=0.0025, tau=0.001):

        self.target_net = self.create_net(state_dim, action_dim).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.action_size = action_dim
        self.gamma = gamma
        self.loss = nn.SmoothL1Loss()
        self.t = 1
        self.tau = tau
        self.epsilon = lambda t: 0.001 + (10 - 0.01) * np.exp(-0.001 * t)

    def create_net(self, s_dim, a_dim, duelling=True):
        if duelling:
            net = DuellingNet(s_dim, a_dim)
        else:
            net = Net(s_dim, a_dim)
        return net

    def get_action(self, state):
        self.t += 1
        if np.random.uniform() > self.epsilon(self.t):
            with torch.no_grad():
                actions = self.policy_net(torch.Tensor(state).to(DEVICE)).cpu().numpy()
            return np.argmax(actions)
        else:
            return torch.tensor(np.random.choice(self.action_size)).numpy()

    def update(self, batch_sample):

        state, action, reward, next_state, done = batch_sample
        states = torch.Tensor(state).to(DEVICE)

        actions = torch.tensor(action, dtype=torch.long).unsqueeze(1).to(DEVICE)

        state_qs = self.policy_net(states).gather(1, actions)

        non_final_states = torch.Tensor(next_state).to(DEVICE)

        next_state_values = self.target_net(non_final_states).max(1)[0].detach()
        next_state_values[done] = 0

        expected_qs = next_state_values * self.gamma + torch.Tensor(reward).to(DEVICE)

        loss = self.loss(state_qs, expected_qs.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        #         for param in self.policy_net.parameters():
        #             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNet, self).__init__()
        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.a = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        l = self.layer(obs)
        out = self.a(l)
        prefs = F.softmax(out, dim=-1)

        return prefs

class ActorCritic:
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.001):

        nets = self.create_net(state_dim, action_dim)
        self.alpha = alpha
        self.policy_net, self.value_net = nets[0].to(DEVICE), nets[1].to(DEVICE)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=alpha)
        self.action_size = action_dim
        self.gamma = gamma
        self.loss = nn.MSELoss()

    def create_net(self, s_dim, a_dim):

        pnet = PolicyNet(s_dim, a_dim)
        vnet = Net(s_dim, action_dim=1)

        return pnet, vnet

    def get_action(self, state):
        """Softmax Policy"""
        with torch.no_grad():
            preferences = self.policy_net(torch.Tensor(state).to(DEVICE))
            action_dist = torch.distributions.Categorical(preferences)
            action = action_dist.sample()
        return action.item()

    def update(self, batch_sample):

        state, action, reward, next_state, done = batch_sample
        states = torch.Tensor(state).to(DEVICE)

        actions = torch.Tensor(action).unsqueeze(1).to(DEVICE)

        state_vs = self.value_net(states)

        non_final_states = torch.Tensor(next_state).to(DEVICE)
        next_state_values = self.value_net(non_final_states).detach()
        next_state_values[done] = 0
        next_state_values = next_state_values.to(DEVICE)
        td_target = torch.Tensor(reward).to(DEVICE) + self.gamma * next_state_values.squeeze(1)

        delta = (td_target - state_vs.squeeze(1)).detach()

        preferences = self.policy_net(states)
        action_dist = torch.distributions.Categorical(preferences.T)
        log_prob = action_dist.log_prob(actions)

        # policy_loss = (-delta * log_prob.T).mean()
        value_loss = self.loss(td_target, state_vs)

        self.policy_optimizer.zero_grad()
        # policy_loss.backward(retain_graph=True)
        # self.policy_optimizer.step()
        with torch.no_grad():
            for i, p in enumerate(self.policy_net.parameters()):
                new_val = p + self.alpha * delta * p.grad
                p.copy_(new_val)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


