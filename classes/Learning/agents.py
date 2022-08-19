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


def numpy_to_cuda(numpy_array):
    return torch.from_numpy(numpy_array).float().to(DEVICE)


class DQN:
    """DQN implementation with epsilon greedy actions selection"""

    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.00263, tau=0.11, rho=0.60, epsilon=1., polyak=False,
                 decay=0.5, step_decay=50000):

        # create simple networks that output Q-values, both target and policy are identical
        self.target_net = self.create_net(state_dim, action_dim, duelling=False).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim, duelling=False).to(DEVICE)

        # We use the Adam optimizer
        self.lr = lr
        self.decay = decay
        self.step_decay = step_decay
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.decay, step_size=step_decay)
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
        self.epsilon = lambda t: 0.01 + epsilon / (t ** self.rho)

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
    def get_action(self, state: np.array, testing=False) -> np.array:
        """We select actions according to epsilon-greedy policy"""
        self.t += 1
        if np.random.uniform() > self.epsilon(self.t) or testing:
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

    @torch.no_grad()
    def next_state_value_estimation(self, next_states, done):
        """Function to define the value of the next state, makes inheritance cleaner"""
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

    def __init__(self, state_dim, action_dim, lr=0.00765, rho=0.76, tau=0.55, **kwargs):
        super(DuelDDQN, self).__init__(state_dim, action_dim, lr=lr, rho=rho, tau=tau, **kwargs)
        # create duelling networks that output Q-values, both target and policy are identical
        self.target_net = self.create_net(state_dim, action_dim, duelling=True).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim, duelling=True).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.decay, step_size=self.step_decay)

    @torch.no_grad()
    def next_state_value_estimation(self, next_states, done):
        """next state value estimation is different for DDQN,
        decouples action selection and evaluation for reduced estimation bias"""
        # find max valued action with policy net
        max_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
        # estimate value of best action with target net
        next_state_values = self.target_net(next_states).gather(1, max_actions)
        next_state_values[done] = 0
        return next_state_values

    def __str__(self):
        return "DuelDDQN"


class A2C:
    """Implementation of the Advantage Actor Critic with entropy regularisation"""

    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=0.0241, lamda=0.161,
                 lr_actor=0.00989, lr_critic=0.00598, step_decay=50000, decay=0.5,
                 max_grad_norm=100):

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.step_decay = step_decay
        self.decay = decay

        # create networks and optimizers
        self.actor, self.critic = self.create_net(state_dim, action_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # create schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optim, gamma=self.decay,
                                                               step_size=step_decay)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optim, gamma=self.decay,
                                                                step_size=step_decay)

        self.action_size = action_dim
        self.gamma = gamma
        self.lamda = lamda

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

        return policy_loss, value_loss

    def policy_loss(self, advantages, new_log_probs, log_probs, entropies):
        """A2C on-policy loss"""
        loss = (-advantages.detach() * new_log_probs - self.epsilon * entropies).mean()

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
    def get_action(self, state, testing=True):
        """For plotting trajectories"""
        preferences = self.actor(numpy_to_cuda(state))
        if testing:
            action = torch.argmax(preferences)
        else:
            probs = Categorical(logits=preferences)
            action = probs.sample()
        return action.item()

    def __str__(self):
        return "A2C"


class PPO(A2C):
    def __init__(self, *args, clip=0.26, lr_actor=0.000713, lr_critic=0.00953, epsilon=0.0213, lamda=0.81,
                 max_grad_norm=100, **kwargs):
        super(PPO, self).__init__(*args, lr_actor=lr_actor, lr_critic=lr_critic, epsilon=epsilon, lamda=lamda,
                                  max_grad_norm=max_grad_norm, **kwargs)
        self.clip = clip

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor, eps=1e-5)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-5)

        # create schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optim, gamma=self.decay,
                                                               step_size=self.step_decay)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optim, gamma=self.decay,
                                                                step_size=self.step_decay)

    def policy_loss(self, advantages, new_log_probs, log_probs, entropies):
        # normalisation of advantages at the mini-batch level
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-20)).detach()

        log_ratio = new_log_probs - log_probs
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(min=1. - self.clip, max=1. + self.clip)
        loss = torch.max(-ratio * advantages, -clipped_ratio * advantages).mean() - self.epsilon * entropies.mean()
        return loss

    def __str__(self):
        return "PPO"
