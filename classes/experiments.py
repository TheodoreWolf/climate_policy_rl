try:
    from Envs.AYS.AYS_Environment import noisy_AYS, velocity_AYS, AYS_Environment
    import Learning.agents as ag
    import Learning.utils as utils
except:
    from climate_policy_RL.classes.Envs.AYS.AYS_Environment import noisy_AYS, velocity_AYS, AYS_Environment
    from climate_policy_RL.classes.Learning import agents as ag
    from climate_policy_RL.classes.Learning import utils as utils
import random
import numpy as np
import os
import torch
import wandb
from learn_class import Learning

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Noisy_Learning(Learning):
    def __init__(self, noise=1e-5, **kwargs):
        super(Noisy_Learning, self).__init__(**kwargs)
        self.env = noisy_AYS(noise_strength=noise)
        self.group_name = "Noisy"


class Markov_Learning(Learning):
    def __init__(self, **kwargs):
        super(Markov_Learning, self).__init__(**kwargs)
        self.env = velocity_AYS(**kwargs)
        self.state_dim = len(self.env.observation_space) * 2
        self.group_name = "Markov"


class Simple_Learning(Learning):
    def __init__(self, cost=0.001, **kwargs):
        super(Simple_Learning, self).__init__(reward_type="policy_cost", **kwargs)
        self.env.management_cost = cost
        self.group_name = "Simple"


if __name__ == "__main__":
    # exp = Markov_Learning(max_episodes=1000, verbose=True)
    # exp.set_agent("PPO")
    # exp.learning_loop_rollout(32, 1024)
    # experiment = Simple_Learning( cost=0.00, wandb_save=True,)
    # experiment.set_agent("DuelDDQN")
    # experiment.learning_loop_offline(128, 2**13, per_is=True)
    experiment = Markov_Learning(wandb_save=True, reward_type="PB", verbose=True)
    experiment.set_agent("DuelDDQN")
    experiment.learning_loop_offline(128, 2**13, per_is=True)


