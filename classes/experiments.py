try:
    from Envs.AYS.AYS_Environment import *
    import Learning.agents as ag
    import Learning.utils as utils
    from learn_class import Learn
except:
    from .Envs.AYS.AYS_Environment import *
    from .Learning import agents as ag
    from .Learning import utils as utils
    from .learn_class import Learn
import random
import numpy as np
import os
import torch
import wandb


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Noisy_Learn(Learn):
    def __init__(self, noise=1e-5, **kwargs):
        super(Noisy_Learn, self).__init__(**kwargs)
        self.env = noisy_AYS(noise_strength=noise)
        self.group_name = "Noisy"


class Markov_Learn(Learn):
    def __init__(self, **kwargs):
        super(Markov_Learn, self).__init__(**kwargs)
        self.env = velocity_AYS(**kwargs)
        self.state_dim = len(self.env.observation_space) * 2
        self.group_name = "Markov"


class Simple_Learn(Learn):
    def __init__(self, cost=0.001, **kwargs):
        super(Simple_Learn, self).__init__(reward_type="policy_cost", **kwargs)
        self.env.management_cost = cost
        self.group_name = "Simple"


if __name__ == "__main__":
    # exp = Markov_Learning(max_episodes=1000, verbose=True)
    # exp.set_agent("PPO")
    # exp.learning_loop_rollout(32, 1024)
    # experiment = Simple_Learning( cost=0.00, wandb_save=True,)
    # experiment.set_agent("DuelDDQN")
    # experiment.learning_loop_offline(128, 2**13, per_is=True)
    experiment = Markov_Learn(wandb_save=False, reward_type="PB", verbose=True)
    experiment.set_agent("DuelDDQN")
    experiment.learning_loop_offline(128, 2**13, per_is=True)


