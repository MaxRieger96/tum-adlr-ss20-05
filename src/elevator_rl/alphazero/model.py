from abc import ABC
from abc import abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch.nn import Linear
from torch.nn import Module
from torch.nn.functional import softmax

from elevator_rl.environment.elevator_env import ElevatorEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


class Model(ABC):
    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def get_policy_and_value(self, env: ElevatorEnv) -> Tuple[np.ndarray, np.ndarray]:
        pass


class NNModel(Module, Model):
    def get_policy_and_value(self, env: ElevatorEnv) -> Tuple[np.ndarray, np.ndarray]:
        observation_array = env.get_observation().as_array()
        x = torch.from_numpy(observation_array)
        x = x.to(device).to(torch.float32)
        policy, value = self(x)
        policy = policy.squeeze().cpu().detach().numpy()
        value = value.squeeze().cpu().detach().numpy()
        return policy, value

    def __init__(self, input_dims: int, outputs: int):
        super(NNModel, self).__init__()
        self.policy_l1 = Linear(input_dims, 2 * outputs)
        self.policy_logits = Linear(2 * outputs, outputs)
        self.value_l1 = Linear(input_dims, 30)
        self.value = Linear(30, 1)

    def forward(self, x):
        # TODO
        value_l1 = self.value_l1(x)
        value = self.value(value_l1)
        policy_logits = self.policy_logits(self.policy_l1(x))
        policy = softmax(policy_logits, dim=0)  # TODO check all of this
        policy = policy + EPS
        return policy, value
