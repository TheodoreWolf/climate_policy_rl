import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from . import networks as nets
except:
    import networks as nets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
numpy_to_cuda = lambda numpy_array: torch.from_numpy(numpy_array).float().to(DEVICE)


class DuelDQN:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.0025, tau=0.001, rho=0.5, epsilon=0.1, polyak=False):

        self.target_net = self.create_net(state_dim, action_dim).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.action_size = action_dim
        self.gamma = gamma
        self.loss = nn.MSELoss()
        self.t = 1
        self.tau = tau
        self.rho = rho
        self.counter = 0
        self.polyak = polyak
        #self.epsilon_ = lambda t: 0.001 + (10 - 0.01) * np.exp(-0.001 * t)
        self.epsilon = lambda t: epsilon * 1/(t**self.rho)

    @staticmethod
    def create_net(s_dim, a_dim, duelling=True):
        if duelling:
            net = nets.DuellingNet(s_dim, a_dim)
        else:
            net = nets.Net(s_dim, a_dim)
        return net

    @torch.no_grad()
    def get_action(self, state):
        self.t += 1
        if np.random.uniform() > self.epsilon(self.t):
            actions = self.policy_net(torch.Tensor(state).to(DEVICE)).cpu().numpy()
            return np.argmax(actions)
        else:
            return torch.tensor(np.random.choice(self.action_size)).numpy()

    def update(self, batch_sample, weights=None):

        state, action, reward, next_state, done = batch_sample

        states = numpy_to_cuda(state)
        actions = numpy_to_cuda(action).type(torch.int64).unsqueeze(1)
        next_states = numpy_to_cuda(next_state)
        rewards = numpy_to_cuda(reward)

        state_qs = self.policy_net(states).gather(1, actions)
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        next_state_values[done] = 0

        expected_qs = next_state_values * self.gamma + rewards
        if weights is not None:
            weights = numpy_to_cuda(weights)
            loss = ((expected_qs.unsqueeze(1)-state_qs)**2 * weights).mean()
        else:
            loss = self.loss(state_qs, expected_qs.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.counter += 1
        if not self.polyak and self.counter >= 1/self.tau:
            self.update_nets()
            self.counter = 0
        else:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        return loss, (state_qs-expected_qs.unsqueeze(1)).detach()

    def online_update(self, time_step):
        raise NotImplementedError

    def update_nets(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DuelDDQN(DuelDQN):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(DuelDDQN, self).__init__(state_dim, action_dim, **kwargs)

    def update(self, batch_sample, weights=None):

        state, action, reward, next_state, done = batch_sample

        states = numpy_to_cuda(state)
        actions = numpy_to_cuda(action).type(torch.int64).unsqueeze(1)
        next_states = numpy_to_cuda(next_state)
        rewards = numpy_to_cuda(reward)

        state_qs = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_state_values = self.target_net(next_states).gather(1, max_actions)
            next_state_values[done] = 0

        expected_qs = next_state_values * self.gamma + rewards.unsqueeze(1)
        if weights is not None:
            weights = numpy_to_cuda(weights)
            loss = ((expected_qs-state_qs).pow(2) * weights).mean()
        else:
            loss = self.loss(state_qs, expected_qs.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.counter += 1
        if not self.polyak and self.counter >= 1 / self.tau:
            self.update_nets()
            self.counter = 0
        else:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        return loss, (state_qs-expected_qs).detach()


class A2C:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001, epsilon=0.01):
        self.lr = lr
        self.epsilon = epsilon
        self.ac_net = nets.DualACNET(state_dim, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=self.lr)
        self.action_size = action_dim
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def create_net(s_dim, a_dim):
        pnet = nets.PolicyNet(s_dim, a_dim).to(DEVICE)
        vnet = nets.Net(s_dim, action_dim=1).to(DEVICE)

        return pnet, vnet

    @torch.no_grad()
    def get_action(self, state):
        """Softmax Policy"""
        preferences = self.ac_net(numpy_to_cuda(state))[1]
        action_dist = Categorical(preferences)
        action = action_dist.sample()
        return action.item()

    def online_update(self, time_step, I):
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

        policy_loss = I * -self.policy_loss(advantage, log_prob, policy)
        value_loss = self.mse_loss(state_vs, td_target)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def policy_loss(self, advantage, log_prob, policy, weights=None):
        if weights is not None:
            return (advantage.detach() * log_prob * weights).mean() - self.epsilon * Categorical(policy).entropy().mean()
        else:
            return (advantage.detach() * log_prob).mean() - self.epsilon * Categorical(policy).entropy().mean()


class PPO(A2C):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(PPO, self).__init__(state_dim, action_dim, **kwargs)

    def policy_loss(self, advantage, log_prob, _, weights=None):
        ratio = (log_prob - log_prob.detach()).exp()
        clipped_ratio = ratio.clamp(min=1.-self.epsilon, max=1.+self.epsilon)
        loss = torch.min(ratio*advantage, clipped_ratio*advantage)
        if weights is not None:
            return (loss*weights).mean()
        else:
            return loss.mean()


class A2Csplit(A2C):
    def __init__(self, state_dim, action_dim, critic_gamma=0.9, actor_gamma=0.9, tau=1000, lr_critic=3e-4, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.t = 0
        self.tau = tau
        self.actor, self.critic = self.create_net(state_dim, action_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optim, gamma=actor_gamma)
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optim, gamma=critic_gamma)

    @torch.no_grad()
    def get_action(self, state):
        """Softmax Policy"""
        preferences = self.actor(numpy_to_cuda(state))
        action_dist = Categorical(preferences)
        action = action_dist.sample()
        return action.item()

    def online_update(self, time_step, I):
        state, action, reward, next_state, done = time_step

        state_t = numpy_to_cuda(state)
        next_state_t = numpy_to_cuda(next_state)

        state_vs = self.critic(state_t)
        next_state_values = self.critic(next_state_t).detach()

        next_state_values[done] = 0

        td_target = reward + self.gamma * next_state_values
        advantage = td_target - state_vs

        policy = self.actor(state_t)
        policy_dist = torch.distributions.Categorical(policy)
        log_prob = policy_dist.log_prob(torch.tensor([action]).to(DEVICE))

        policy_loss = I * -self.policy_loss(advantage, log_prob, policy)
        value_loss = self.mse_loss(state_vs, td_target)

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        self.t += 1
        if self.t >= self.tau:
            self.critic_scheduler.step()
            self.actor_scheduler.step()
        return policy_loss+value_loss


class PPOsplit(A2Csplit):
    def __init__(self, state_dim, action_dim, clip=0.2, critic_gamma=0.9, actor_gamma=0.9,
                 max_grad_norm=20, lr_critic=3e-4, critic_param=0.5, labda=0.95, **kwargs):
        super(PPOsplit, self).__init__(state_dim, action_dim, **kwargs)
        self.t = 0
        self.clip = clip
        self.actor, self.critic = self.create_net(state_dim, action_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optim, gamma=actor_gamma)
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optim, gamma=critic_gamma)
        self.labda = labda
        self.max_grad_norm = max_grad_norm
        self.critic_param = critic_param

    def get_action_and_value(self, state, action=None):
        preferences = self.actor(state)
        probs = Categorical(logits=preferences)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)

    def online_update(self, time_step, I=1):
        state, action, reward, next_state, done = time_step

        state_t = numpy_to_cuda(state)
        next_state_t = numpy_to_cuda(next_state)

        state_vs = self.critic(state_t)
        next_state_values = self.critic(next_state_t).detach()

        next_state_values[done] = 0

        td_target = reward + self.gamma * next_state_values
        advantage = td_target - state_vs

        policy = self.actor(state_t)
        policy_dist = torch.distributions.Categorical(logits=policy)
        log_prob = policy_dist.log_prob(torch.tensor([action]).to(DEVICE))

        policy_loss = I * -self.policy_loss(advantage.detach(), log_prob, policy)
        value_loss = self.mse_loss(state_vs, td_target)

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        self.t += 1
        if self.t >= self.tau:
            self.critic_scheduler.step()
            self.actor_scheduler.step()
        return policy_loss + value_loss

    def update(self, batch_sample):
        # we receive a minibatch of stuff
        states, actions, values, rewards, dones, logprobs, advantages, returns = batch_sample # torch cuda tensors

        _, new_log_probs, entropies, new_values = self.get_action_and_value(states, actions.long())

        log_ratio = new_log_probs - logprobs
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(min=1.-self.clip, max=1.+self.clip)
        advantages = ((advantages-advantages.mean()) / (advantages.std()+1e-8)).detach()

        value_loss = self.mse_loss(new_values.squeeze(1), returns.detach()) * self.critic_param
        policy_loss = torch.max(-ratio*advantages, -clipped_ratio*advantages).mean() - self.epsilon * entropies.mean()

        self.critic_optim.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

    def ppo_loss(self, advantage, log_prob,):
        ratio = (log_prob - log_prob.detach()).exp()
        clipped_ratio = ratio.clamp(min=1.-self.epsilon, max=1.+self.epsilon)
        loss = torch.min(ratio*advantage, clipped_ratio*advantage)
        return loss.mean()
