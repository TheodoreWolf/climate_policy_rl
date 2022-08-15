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


class noisy_Learning(Learning):
    def __init__(self, noise=1e-5, **kwargs):
        super(noisy_Learning, self).__init__(**kwargs)
        self.env = noisy_AYS(noise_strength=noise)


class Markov_Learning(Learning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env = velocity_AYS()
        self.state_dim = len(self.env.observation_space) * 2


if __name__ == "__main__":
    exp = Markov_Learning(max_episodes=1000, verbose=True)
    exp.set_agent("PPO")
    exp.learning_loop_rollout(32, 1024)
