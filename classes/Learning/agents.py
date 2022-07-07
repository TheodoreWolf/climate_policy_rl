import numpy as np
import torch
import torch.nn as nn
try:
    from . import networks as nets
except:
    import networks as nets
from torch.distributions import Categorical

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
numpy_to_cuda = lambda numpy_array: torch.from_numpy(numpy_array).float().to(DEVICE)


class DuellingDQN:
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

    @staticmethod
    def create_net(s_dim, a_dim, duelling=True):
        if duelling:
            net = nets.DuellingNet(s_dim, a_dim)
        else:
            net = nets.Net(s_dim, a_dim)
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
        non_final_states = torch.Tensor(next_state).to(DEVICE)

        state_qs = self.policy_net(states).gather(1, actions)
        next_state_values = self.target_net(non_final_states).max(1)[0].detach()
        next_state_values[done] = 0

        expected_qs = next_state_values * self.gamma + torch.Tensor(reward).to(DEVICE)

        loss = self.loss(state_qs, expected_qs.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy_net.parameters(), max_norm=2)
        self.optimizer.step()

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        return loss

    def online_update(self, time_step):
        raise NotImplementedError


class ActorCritic:
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.001, beta=0.01):
        self.alpha = alpha
        self.beta = beta
        self.ac_net = nets.DualACNET(state_dim, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=self.alpha)
        self.action_size = action_dim
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def create_net(s_dim, a_dim):
        pnet = nets.PolicyNet(s_dim, a_dim)
        vnet = nets.Net(s_dim, action_dim=1)

        return pnet, vnet

    def get_action(self, state):
        """Softmax Policy"""
        preferences = self.ac_net(numpy_to_cuda(state))[1]
        action_dist = Categorical(preferences)
        action = action_dist.sample()
        return action.item()

    def update(self, batch_sample):
        state, action, reward, next_state, done = batch_sample

        states = numpy_to_cuda(state)
        next_states = numpy_to_cuda(next_state)
        actions = numpy_to_cuda(action)
        rewards = numpy_to_cuda(reward)

        state_vs = self.ac_net(states)[0]
        next_state_values = self.ac_net(next_states)[0].detach()

        next_state_values[done] = 0

        td_target = rewards + self.gamma * next_state_values
        advantage = td_target - state_vs
        print(td_target.shape, td_target)
        policy = self.ac_net(states)[1]
        policy_dist = torch.distributions.Categorical(policy)
        log_prob = policy_dist.log_prob(actions)

        policy_loss = self.policy_loss(advantage, log_prob, policy)
        value_loss = self.mse_loss(state_vs, td_target)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def online_update(self, time_step):

        state, action, reward, next_state, done = time_step

        state_t = numpy_to_cuda(state)
        next_state_t = numpy_to_cuda(next_state)

        state_vs = self.ac_net(state_t)[0]
        next_state_values = self.ac_net(next_state_t)[0].detach()

        next_state_values[done] = 0

        td_target = reward + self.gamma * next_state_values
        advantage = td_target - state_vs

        policy = self.ac_net(state_t)[1]
        policy_dist = torch.distributions.Categorical(policy)
        log_prob = policy_dist.log_prob(torch.tensor([action]).to(DEVICE))

        policy_loss = self.policy_loss(advantage, log_prob, policy)
        value_loss = self.mse_loss(state_vs, td_target)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def policy_loss(self, advantage, log_prob, policy):
        return (-advantage.detach() * log_prob).mean() - self.beta * Categorical(policy).entropy().mean()


class PPO(ActorCritic):
    def __init__(self, state_dim, action_dim, alpha=0.001, gamma=0.99, epsilon=0.001):
        super().__init__(state_dim, action_dim, gamma=0.99, alpha=0.001)

        self.epsilon = epsilon

    def policy_loss(self, advantage, log_prob, _):
        loss = -torch.min((log_prob - log_prob.detach()).exp(), 1+torch.sign(advantage.detach())*self.epsilon) \
               * advantage.detach()
        return loss.mean()


class TRPO(ActorCritic):
    def __init__(self, state_dim, action_dim, alpha=0.001, gamma=0.99, beta=0.1):
        super().__init__(state_dim, action_dim, gamma=0.99, alpha=0.001)

        self.beta = beta

    def policy_loss(self, advantage, log_prob, policy):
        policy_dist = Categorical(policy)
        loss = -(log_prob-log_prob.detach())*advantage.detach() - self.beta*(policy*(policy.log()-policy.detach().log()))
        return loss.mean()



