# @Theodore Wolf

import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output


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


def plot(frame_idx, rewards):
    """For tracking experiment progress"""
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


def plot_test_trajectory(env, agent, max_steps=600):
    """To plot trajectories of the agent"""
    state = env.reset_for_state()
    learning_progress = []
    for step in range(max_steps):
        list_state = env.get_plot_state_list()

        # take recommended action
        action = agent.get_action(state)

        # Do the new chosen action in Environment
        new_state, reward, done = env.step(action)

        learning_progress.append([list_state, action, reward])

        state = new_state
        if done:
            break

    env.plot_run(learning_progress)

    return env._which_final_state()


class PER_IS_ReplayBuffer:
    """
    Adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations
    """
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1.
        self.data = {
            'obs': np.zeros(shape=(capacity, 3), dtype=np.float64),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, 3), dtype=np.float64),
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
            if self.priority_sum[idx*2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx*2]
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

        prob_min = self._min()/self._sum()
        max_weight = (prob_min*self.size)**(-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx+self.capacity]/self._sum()
            weight = (prob*self.size)**(-beta)
            samples['weights'][i] = weight/max_weight

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
