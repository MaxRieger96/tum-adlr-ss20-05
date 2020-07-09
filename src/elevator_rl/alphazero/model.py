from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch import relu
from torch.nn import Linear, BatchNorm1d
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
        self.to(device)

        # get observation
        observation_arrays = env.get_observation().as_array()
        house_obs, current_elevator_obs, other_elevators_obs = observation_arrays

        # move observations to torch tensors on gpu
        house_obs = (
            torch.from_numpy(house_obs).to(device).to(torch.float32).unsqueeze(dim=0)
        )
        current_elevator_obs = (
            torch.from_numpy(current_elevator_obs)
            .to(device)
            .to(torch.float32)
            .unsqueeze(dim=0)
        )
        other_elevators_obs = [
            torch.from_numpy(e).to(device).to(torch.float32).unsqueeze(dim=0)
            for e in other_elevators_obs
        ]

        # evaluate net
        policy, value = self.forward(
            house_obs, current_elevator_obs, other_elevators_obs,
        )

        # move policy and value to numpy arrays on cpu
        policy = policy.squeeze().cpu().detach().numpy()
        value = value.squeeze().cpu().detach().numpy()
        return policy, value

    def __init__(
        self,
        house_observation_dims: int,
        elevator_observation_dims: int,
        policy_dims: int,
    ):
        # TODO adjust model for new observation format
        super(NNModel, self).__init__()
        self.elevator_encoder_l1 = Linear(elevator_observation_dims, 64)
        self.elevator_encoder_bn1 = BatchNorm1d(64)
        self.elevator_encoder_l2 = Linear(64, 64)
        self.elevator_encoder_bn2 = BatchNorm1d(64)

        self.house_encoder_l1 = Linear(house_observation_dims, 64)
        self.house_encoder_bn1 = BatchNorm1d(64)
        self.house_encoder_l2 = Linear(64, 64)
        self.house_encoder_bn2 = BatchNorm1d(64)

        self.state_transform_l1 = Linear(64 + 64 + 64, 128)
        self.state_transform_bn1 = BatchNorm1d(128)
        self.state_transform_l2 = Linear(128, 64)
        self.state_transform_bn2 = BatchNorm1d(64)

        self.policy_output_l1 = Linear(64, policy_dims)
        self.value_output_l1 = Linear(64, 1)

    def encode_house(self, house_observation: Tensor) -> Tensor:
        house_observation_encoding = relu(
            self.house_encoder_bn1(self.house_encoder_l1(house_observation))
        )
        house_observation_encoding = relu(
            self.house_encoder_bn2(self.house_encoder_l2(house_observation_encoding))
        )
        return house_observation_encoding

    def encode_elevator(self, elevator_observation) -> Tensor:
        elevator_observation_encoding = relu(
            self.elevator_encoder_bn1(self.elevator_encoder_l1(elevator_observation))
        )
        elevator_observation_encoding = relu(
            self.elevator_encoder_bn2(
                self.elevator_encoder_l2(elevator_observation_encoding)
            )
        )
        return elevator_observation_encoding

    def forward(
        self,
        house_observation: Tensor,
        current_elevator_observation: Tensor,
        other_elevator_observations: List[Tensor],
    ):
        # encode current elevator, house, and other elevators
        current_elevator_encoding = self.encode_elevator(current_elevator_observation)
        house_encoding = self.encode_house(house_observation)
        other_elevators_encoding = [
            self.encode_elevator(e) for e in other_elevator_observations
        ]

        # combine encodings to single vector
        if len(other_elevators_encoding)>0:
            other_elevators_encoding = torch.stack(
                [e for e in other_elevators_encoding], dim=1
            ).sum(dim=1, keepdim=False)
        else:
            other_elevators_encoding = torch.zeros_like(current_elevator_encoding)
        combined_state = torch.cat(
            [current_elevator_encoding, house_encoding, other_elevators_encoding],
            dim=1,
        )

        # transform state encoding vector
        combined_state = relu(
            self.state_transform_bn1(self.state_transform_l1(combined_state))
        )
        combined_state = relu(
            self.state_transform_bn2(self.state_transform_l2(combined_state))
        )

        # output policy
        policy_logits = self.policy_output_l1(combined_state)
        policy = softmax(policy_logits, dim=1)  # TODO check all of this
        policy = policy + EPS

        # output value
        value = self.value_output_l1(combined_state)
        value = torch.tanh(value)
        return policy, value
