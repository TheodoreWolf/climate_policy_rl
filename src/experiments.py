import os

import torch

from envs.AYS.AYS_Environment import *
from learn_class import Learn


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PB_Learn(Learn):
    def __init__(self, **kwargs):
        super(PB_Learn, self).__init__(**kwargs)


class Noisy_Learn(Learn):
    def __init__(self, noise=1e-5, periodic_increase=500, markov=False, fixed_test=False, **kwargs):
        super(Noisy_Learn, self).__init__(**kwargs)
        if not markov:
            self.env = noisy_AYS(noise_strength=noise, periodic_increase=periodic_increase, fixed=fixed_test, **kwargs)
            self.group_name = "Noisy"
        else:
            self.env = Noisy_Markov(noise_strength=noise, fixed=fixed_test, **kwargs)
            self.state_dim = len(self.env.observation_space) * 2
            self.group_name = "Noisy_Markov"


class Markov_Learn(Learn):
    def __init__(self, **kwargs):
        super(Markov_Learn, self).__init__(**kwargs)
        self.env = velocity_AYS(**kwargs)
        self.state_dim = len(self.env.observation_space) * 2
        self.group_name = "Markov"


class Simple_Learn(Learn):
    def __init__(self, cost=0.001, **kwargs):
        super(Simple_Learn, self).__init__(reward_type="simple", **kwargs)
        self.env.management_cost = cost
        self.group_name = "Simple"


if __name__ == "__main__":
    experiment = Markov_Learn(wandb_save=False, reward_type="PB", verbose=True)
    experiment.set_agent("DuelDDQN")
    experiment.learning_loop_offline(128, 2**13, per_is=True)


