from abc import ABC

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

class DQN:
    """DQN implementation with epsilon greedy actions selection"""

    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.0015, tau=0.04, rho=0.8, epsilon=0.3, polyak=False):

        # create simple networks that output Q-values, both target and policy are identical
        self.target_net = self.create_net(state_dim, action_dim, duelling=False).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim, duelling=False).to(DEVICE)

        # We use the Adam optimizer
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.action_size = action_dim
        self.gamma = gamma

        # Squared loss, as it is easier to work with than the Huber loss
        self.loss = nn.MSELoss()

        # For copying networks
        self.tau = tau
        self.counter = 0
        self.polyak = polyak

        # We decay epsilon according to the following formula
        self.t = 1
        self.rho = rho
        self.epsilon = lambda t: epsilon * 1 / (t ** self.rho)

    @staticmethod
    def create_net(s_dim, a_dim, duelling):
        """We create action-out networks that can be duelling or not,
         Duelling is more stable to optimisation"""
        if duelling:
            net = nets.DuellingNet(s_dim, a_dim)
        else:
            net = nets.Net(s_dim, a_dim)
        return net

    @torch.no_grad()
    def get_action(self, state: np.array) -> torch.tensor:
        """We select actions according to epsilon-greedy policy"""
        self.t += 1
        if np.random.uniform() > self.epsilon(self.t):
            q_values = self.policy_net(torch.Tensor(state).to(DEVICE)).cpu().numpy()
            return np.argmax(q_values)
        else:
            return np.random.choice(self.action_size)

    def update(self, batch_sample, weights=None):
        """To update our networks"""
        # Unpack batch: 5-tuple
        state, action, reward, next_state, done = batch_sample

        # convert to torch.cuda
        states = numpy_to_cuda(state)
        actions = numpy_to_cuda(action).type(torch.int64).unsqueeze(1)
        next_states = numpy_to_cuda(next_state)
        rewards = numpy_to_cuda(reward)

        # get the Q-values of the actions at time t
        state_qs = self.policy_net(states).gather(1, actions).squeeze(1)

        # get the max Q-values at t+1 from the target network
        next_state_values = self.next_state_value_estimation(next_states, done)

        # target: y_t = r_t + gamma * max[Q(s,a)]
        targets = (rewards + self.gamma * next_state_values.squeeze(1))

        # if we have weights from importance sampling
        if weights is not None:
            weights = numpy_to_cuda(weights)
            loss = ((targets - state_qs).pow(2) * weights).mean()
        # otherwise we use the standard MSE loss
        else:
            loss = self.loss(state_qs, targets)

        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # to copy the policy parameters to the target network
        self.copy_nets()
        # we return the loss for logging and the TDs for Prioritised Experience Replay
        return loss, (state_qs - targets).detach()

    def next_state_value_estimation(self, next_states, done):
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        # the value of a state after a terminal state is 0
        next_state_values[done] = 0
        return next_state_values.unsqueeze(1)

    def copy_nets(self):
        """Copies the parameters from the policy network to the target network, either all at once or incrementally."""
        self.counter += 1
        if not self.polyak and self.counter >= 1 / self.tau:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.counter = 0
        else:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def __str__(self):
        return "DQN"


class DuelDDQN(DQN):
    """Implementation of DuelDDQN, inspired by RAINBOW"""
    def __init__(self, state_dim, action_dim, **kwargs):
        super(DuelDDQN, self).__init__(state_dim, action_dim, **kwargs)
        # create duelling networks that output Q-values, both target and policy are identical
        self.target_net = self.create_net(state_dim, action_dim, duelling=True).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim, duelling=True).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def next_state_value_estimation(self, next_states, done):
        # next state value estimation is different for DDQN
        with torch.no_grad():
            # find max action with policy net
            max_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            # estimate value of best action with target net
            next_state_values = self.target_net(next_states).gather(1, max_actions)
            next_state_values[done] = 0
            return next_state_values

    def __str__(self):
        return "DuelDDQN"


class A2C:
    """Implementation of the Advantage Actor Critic with entropy regularisation"""

    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=0.001, lamda=0.85,
                 lr_actor=3e-4, lr_critic=3e-4,
                 actor_decay=0.9, critic_decay=0.9,
                 max_grad_norm=100, critic_param=1.):
        # create networks and optimizers
        self.actor, self.critic = self.create_net(state_dim, action_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        # create schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optim, gamma=actor_decay)
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optim, gamma=critic_decay)

        self.action_size = action_dim
        self.gamma = gamma
        self.lamda = lamda
        self.critic_param = critic_param

        self.value_loss = nn.MSELoss()
        self.max_grad_norm = max_grad_norm

        # temperature parameter that controls the entropy regularisation, analogous to exploration
        self.epsilon = epsilon

    @staticmethod
    def create_net(s_dim, a_dim):
        """ create both policy and value networks: actor and critic"""
        actor_net = nets.PolicyNet(s_dim, a_dim).to(DEVICE)
        critic_net = nets.Net(s_dim, action_dim=1).to(DEVICE)

        return actor_net, critic_net

    def get_action_and_value(self, state, action=None):
        preferences = self.actor(state)
        probs = Categorical(logits=preferences)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)

    def update(self, mini_batch_sample):
        """Update with a mini-batch"""
        # unpack mini-batch, already torch.cuda tensors
        states, actions, values, rewards, dones, log_probs, advantages, returns = mini_batch_sample

        # normalisation of advantages at the mini-batch level
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).detach()

        # get some gradients
        _, new_log_probs, entropies, new_values = self.get_action_and_value(states, actions.long())

        # backpropagation
        value_loss = self.value_loss(new_values.squeeze(1), returns.detach())
        policy_loss = self.policy_loss(advantages, new_log_probs, log_probs, entropies)

        self.critic_optim.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

    def policy_loss(self, advantages, new_log_probs, log_probs, entropies):
        """A2C on-policy loss"""
        loss = (-advantages * new_log_probs - self.epsilon * entropies).mean()

        return loss

    def compute_gae(self, values, dones, rewards, next_value, next_done):
        """Compute the Generalised advantages at each time step (called by the experiment class)"""
        last_adv = 0
        buffer_size = len(rewards)
        advantages = torch.zeros_like(rewards).to(DEVICE)
        for t in reversed(range(buffer_size)):
            if t == buffer_size - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = last_adv = delta + self.gamma * self.lamda * nextnonterminal * last_adv
        returns = advantages + values
        return returns, advantages

    @torch.no_grad()
    def get_action(self, state):
        """For plotting trajectories"""
        preferences = self.actor(numpy_to_cuda(state))
        probs = Categorical(logits=preferences)
        action = probs.sample()
        return action

    def __str__(self):
        return "A2C"


class PPO(A2C):
    def __init__(self, *args, clip=0.2, **kwargs):
        super(PPO, self).__init__(*args, **kwargs)
        self.clip = clip

    def policy_loss(self, advantages, new_log_probs, log_probs, entropies):
        log_ratio = new_log_probs - log_probs
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(min=1. - self.clip, max=1. + self.clip)
        loss = torch.max(-ratio * advantages, -clipped_ratio * advantages).mean() - self.epsilon * entropies.mean()
        return loss

    def __str__(self):
        return "PPO"

# class A2C2:
#     def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001, epsilon=0.01):
#         self.lr = lr
#         self.epsilon = epsilon
#         self.ac_net = nets.DualACNET(state_dim, action_dim).to(DEVICE)
#         self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=self.lr)
#         self.action_size = action_dim
#         self.gamma = gamma
#         self.mse_loss = nn.MSELoss()
#
#     @staticmethod
#     def create_net(s_dim, a_dim):
#         pnet = nets.PolicyNet(s_dim, a_dim).to(DEVICE)
#         vnet = nets.Net(s_dim, action_dim=1).to(DEVICE)
#
#         return pnet, vnet
#
#     @torch.no_grad()
#     def get_action(self, state):
#         """Softmax Policy"""
#         preferences = self.ac_net(numpy_to_cuda(state))[1]
#         action_dist = Categorical(preferences)
#         action = action_dist.sample()
#         return action.item()
#
#     def online_update(self, time_step, I):
#         state, action, reward, next_state, done = time_step
#
#         state_t = numpy_to_cuda(state)
#         next_state_t = numpy_to_cuda(next_state)
#
#         state_vs = self.ac_net(state_t)[0]
#         next_state_values = self.ac_net(next_state_t)[0].detach()
#
#         next_state_values[done] = 0
#
#         td_target = reward + self.gamma * next_state_values
#         advantage = td_target - state_vs
#
#         policy = self.ac_net(state_t)[1]
#         policy_dist = torch.distributions.Categorical(policy)
#         log_prob = policy_dist.log_prob(torch.tensor([action]).to(DEVICE))
#
#         policy_loss = I * -self.policy_loss(advantage, log_prob, policy)
#         value_loss = self.mse_loss(state_vs, td_target)
#         loss = policy_loss + value_loss
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return loss
#
#     def policy_loss(self, advantage, log_prob, policy, weights=None):
#         if weights is not None:
#             return (advantage.detach() * log_prob * weights).mean() - self.epsilon * Categorical(
#                 policy).entropy().mean()
#         else:
#             return (advantage.detach() * log_prob).mean() - self.epsilon * Categorical(policy).entropy().mean()


# class PPO(A2C):
#     def __init__(self, state_dim, action_dim, **kwargs):
#         super(PPO, self).__init__(state_dim, action_dim, **kwargs)
#
#     def policy_loss(self, advantage, log_prob, _, weights=None):
#         ratio = (log_prob - log_prob.detach()).exp()
#         clipped_ratio = ratio.clamp(min=1. - self.epsilon, max=1. + self.epsilon)
#         loss = torch.min(ratio * advantage, clipped_ratio * advantage)
#         if weights is not None:
#             return (loss * weights).mean()
#         else:
#             return loss.mean()
#
#
# class A2Csplit(A2C):
#     def __init__(self, state_dim, action_dim, critic_gamma=0.9, actor_gamma=0.9, tau=1000, lr_critic=3e-4, **kwargs):
#         super().__init__(state_dim, action_dim, **kwargs)
#         self.t = 0
#         self.tau = tau
#         self.actor, self.critic = self.create_net(state_dim, action_dim)
#         self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
#         self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optim, gamma=actor_gamma)
#         self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optim, gamma=critic_gamma)
#
#     @torch.no_grad()
#     def get_action(self, state):
#         """Softmax Policy"""
#         preferences = self.actor(numpy_to_cuda(state))
#         action_dist = Categorical(preferences)
#         action = action_dist.sample()
#         return action.item()
#
#     def online_update(self, time_step, I):
#         state, action, reward, next_state, done = time_step
#
#         state_t = numpy_to_cuda(state)
#         next_state_t = numpy_to_cuda(next_state)
#
#         state_vs = self.critic(state_t)
#         next_state_values = self.critic(next_state_t).detach()
#
#         next_state_values[done] = 0
#
#         td_target = reward + self.gamma * next_state_values
#         advantage = td_target - state_vs
#
#         policy = self.actor(state_t)
#         policy_dist = torch.distributions.Categorical(policy)
#         log_prob = policy_dist.log_prob(torch.tensor([action]).to(DEVICE))
#
#         policy_loss = I * -self.policy_loss(advantage, log_prob, policy)
#         value_loss = self.mse_loss(state_vs, td_target)
#
#         self.actor_optim.zero_grad()
#         policy_loss.backward()
#         self.actor_optim.step()
#
#         self.critic_optim.zero_grad()
#         value_loss.backward()
#         self.critic_optim.step()
#
#         self.t += 1
#         if self.t >= self.tau:
#             self.critic_scheduler.step()
#             self.actor_scheduler.step()
#         return policy_loss + value_loss


# class PPOsplit(A2C2):
#     def __init__(self, state_dim, action_dim, clip=0.2, critic_gamma=0.9, actor_gamma=0.9,
#                  max_grad_norm=20, lr_critic=3e-4, critic_param=1.5, lamda=0.95, **kwargs):
#         super(PPOsplit, self).__init__(state_dim, action_dim, **kwargs)
#         self.t = 0
#         self.lamda = lamda
#         self.clip = clip
#         self.actor, self.critic = self.create_net(state_dim, action_dim)
#         self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
#         self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
#         self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optim, gamma=actor_gamma)
#         self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optim, gamma=critic_gamma)
#         self.max_grad_norm = max_grad_norm
#         self.critic_param = critic_param
#
#     def get_action_and_value(self, state, action=None):
#         preferences = self.actor(state)
#         probs = Categorical(logits=preferences)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(state)
#
#     def update(self, batch_sample):
#         # we receive a minibatch of stuff
#         states, actions, values, rewards, dones, logprobs, advantages, returns = batch_sample  # torch cuda tensors
#
#         _, new_log_probs, entropies, new_values = self.get_action_and_value(states, actions.long())
#
#         log_ratio = new_log_probs - logprobs
#         ratio = log_ratio.exp()
#         clipped_ratio = ratio.clamp(min=1. - self.clip, max=1. + self.clip)
#         advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).detach()
#
#         value_loss = self.mse_loss(new_values.squeeze(1), returns.detach()) * self.critic_param
#         policy_loss = torch.max(-ratio * advantages,
#                                 -clipped_ratio * advantages).mean() - self.epsilon * entropies.mean()
#
#         self.critic_optim.zero_grad()
#         value_loss.backward()
#         nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
#         self.critic_optim.step()
#
#         self.actor_optim.zero_grad()
#         policy_loss.backward()
#         nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
#         self.actor_optim.step()
#
#     def ppo_loss(self, advantage, log_prob, ):
#         ratio = (log_prob - log_prob.detach()).exp()
#         clipped_ratio = ratio.clamp(min=1. - self.epsilon, max=1. + self.epsilon)
#         loss = torch.min(ratio * advantage, clipped_ratio * advantage)
#         return loss.mean()
#
#     def compute_gae(self, values, dones, rewards, next_value, next_done):
#         """Compute the Generalised advantages at each time step (called by the experiment class)"""
#         last_adv = 0
#         buffer_size = len(rewards)
#         advantages = torch.zeros_like(rewards).to(DEVICE)
#         for t in reversed(range(buffer_size)):
#             if t == buffer_size - 1:
#                 nextnonterminal = 1.0 - next_done
#                 nextvalues = next_value
#             else:
#                 nextnonterminal = 1.0 - dones[t + 1]
#                 nextvalues = values[t + 1]
#             delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
#             advantages[t] = last_adv = delta + self.gamma * self.lamda * nextnonterminal * last_adv
#         returns = advantages + values
#         return returns, advantages
