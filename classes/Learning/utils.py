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


class PERBuffer(ReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)

        self.capacity = capacity
        self.alpha = 0.1
        self.beta = 0.5

    def update_priorities(self,tds, ):
        un_norm_probs = np.abs(tds)**self.alpha
        norm_probs = un_norm_probs/np.sum(un_norm_probs)

    def get_tds(self, agent_model, agent_gamma):
        next_q = agent_model(next_states)
        next_q[done] = 0
        tds = rewards + agent_gamma * next_q - agent_model(states)


def plot(frame_idx, rewards):
    """For tracking experiment progress"""
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def plot_test_trajectory(env, agent, max_steps = 600):
    """To plot trajectories of the agent"""
    state = env.reset_for_state()
    learning_progress = []
    for step in range(max_steps):
        list_state = env.get_plot_state_list()

        # take recommended action
        action = agent.get_action(state)

        # Do the new chosen action in Environment
        new_state, reward, done = env.step(action)

        learning_progress.append([list_state, action, reward])  # TODO Fix for noisy state!

        state = new_state
        if done:
            break

    env.plot_run(learning_progress)

    return env._which_final_state()

