# @Theodore Wolf

import numpy as np
import random
from IPython.display import clear_output
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    """To store experience for uncorrelated learning"""

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


def plot(data_dict):
    """For tracking experiment progress"""
    rewards = data_dict['moving_avg_rewards']
    std = data_dict['moving_std_rewards']
    frame_idx = data_dict['frame_idx']
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    reward = np.array(rewards)
    stds = np.array(std)
    plt.fill_between(np.arange(len(reward)), reward - 0.25 * stds, reward + 0.25 * stds, color='b', alpha=0.1)
    plt.fill_between(np.arange(len(reward)), reward - 0.5 * stds, reward + 0.5 * stds, color='b', alpha=0.1)
    plt.show()


def plot_test_trajectory(env, agent, max_steps=600, test_state=None, fname=None):
    """To plot trajectories of the agent"""
    state = env.reset_for_state(test_state)
    learning_progress = []
    for step in range(max_steps):
        list_state = env.get_plot_state_list()

        # take recommended action
        action = agent.get_action(state, testing=True)

        # Do the new chosen action in Environment
        new_state, reward, done, _ = env.step(action)

        learning_progress.append([list_state, action, reward])

        state = new_state
        if done:
            break

    env.plot_run(learning_progress, fname=fname)

    return env.which_final_state()


class PER_IS_ReplayBuffer:
    """
    Adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations
    """

    def __init__(self, capacity, alpha, state_dim=3):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1.
        self.data = {
            'obs': np.zeros(shape=(capacity, state_dim), dtype=np.float64),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, state_dim), dtype=np.float64),
            'done': np.zeros(shape=capacity, dtype=np.bool)
        }
        self.next_idx = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        idx = self.next_idx
        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['done'][idx] = done

        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_sum[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        return idx - self.capacity

    def sample(self, batch_size, beta):

        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32),
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples['weights'][i] = weight / max_weight

        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        return samples

    def update_priorities(self, indexes, priorities):

        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        return self.capacity == self.size

    def __len__(self):
        return self.size


def feature_importance(agent_net, buffer, n_points, v=False, scalar=False):
    features = ["A", "Y", "S"]
    if v:
        features = ["A", "Y", "S", "dA", "dY", "dS"]

    data = buffer.sample(n_points)[0]

    explainer = shap.DeepExplainer(agent_net,
                                   torch.from_numpy(data).float().to(DEVICE))
    shap_q_values = explainer.shap_values(torch.from_numpy(data).float().to(DEVICE))
    if scalar:
        shap_values = np.array(shap_q_values)
    else:
        shap_values = np.array(np.sum(shap_q_values, axis=0))
    shap.summary_plot(shap_values,
                        features=data,
                        feature_names=features,
                        plot_type='violin', show=False, sort=False)


def plot_end_state_matrix(results):
    t = 1 # alpha value
    size = int(np.sqrt(len(results)))
    cmap = {1: [0., 0., 0., t], 2: [0., 1.0, 0., t], 3: [1.0, 0.1, 0.1, t], 4: [1., 1., 0., t]}
    labels = {1: r'$Black_{FP}$', 2: r'$Green_{FP}$', 3: r'$A_{PB}$', 4: r'$Y_{SF}$'}
    arrayShow = np.array([[cmap[i] for i in j] for j in results.reshape(size, size)])
    patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
    plt.imshow(arrayShow, extent=(0.45, 0.55, 0.55, 0.45))
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1.))
    plt.ylabel("A")
    plt.xlabel("Y")


def plot_action_matrix(results):
    t = 1 # alpha value
    size = int(np.sqrt(len(results)))
    cmap = {0: [1.0, 0.1, 0.1, t], 1: [1., 0.5, 0., t], 2: [0.1, 1., 0.1, t], 3: [0., 0., 1., t]}
    labels = {0: 'Default', 1: 'DG', 2: 'ET', 3: 'DG+ET'}
    arrayShow = np.array([[cmap[i] for i in j] for j in results.reshape(size, size)])
    patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
    plt.imshow(arrayShow, extent=(0.45, 0.55, 0.55, 0.45))
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1.))
    plt.ylabel("A")
    plt.xlabel("Y")
