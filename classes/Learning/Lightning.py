import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import networks as net
import wandb
from utils import ReplayBuffer


class DQNLightning(pl.LightningModule):
    def __init__(self,
                 config: dict,
                 model: nn.Module,
                 env: object,
                 name: str):
        super().__init__()
        self.save_hyperparameters()

        self.network = model
        wandb.watch(self.network)
        self.__name__ = name
        self.env = env
        self.buffer = ReplayBuffer(config['BUFFER_SIZE'])

        wandb.init(project="AYS_learning", entity="climate_policy_optim", name=name, config=config)

    def forward(self, state) -> Tensor:
        return self.network(state)

    def training_step(self, *args, **kwargs) -> float:
        pass




