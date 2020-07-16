from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Tuple

import numpy as np
import torch
from torch import relu
from torch import Tensor
from torch.nn import BatchNorm1d
from torch.nn import Linear
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.nn.functional import softmax
from torch.optim import Adam

from elevator_rl.alphazero.ranked_reward import RankedRewardBuffer
from elevator_rl.alphazero.replay_buffer import ReplayBuffer
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
        other_elevators_obs = np.array(other_elevators_obs)
        other_elevators_obs = (
            torch.from_numpy(other_elevators_obs)
            .to(device)
            .to(torch.float32)
            .unsqueeze(dim=0)
        )

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
        other_elevator_observations: Tensor,
    ):
        # encode current elevator, house, and other elevators
        current_elevator_encoding = self.encode_elevator(current_elevator_observation)
        house_encoding = self.encode_house(house_observation)
        other_elevators_encoding = []
        for elevator_index in range(other_elevator_observations.shape[1]):
            other_elevators_encoding.append(
                self.encode_elevator(other_elevator_observations[:, elevator_index])
            )

        # combine encodings to single vector
        if len(other_elevators_encoding) > 0:
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


def train(
    model: NNModel,
    replay_buffer: ReplayBuffer,
    ranked_reward_buffer: RankedRewardBuffer,
    offset: int,
    config: Dict,
):
    optimizer = Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    batch_count = (
        config["train"]["samples_per_iteration"] // config["train"]["batch_size"]
    )
    model.to(device)
    model.train()
    acc_loss_value = []
    acc_loss_policy = []
    logs = []
    for i in range(batch_count):
        samples = replay_buffer.sample(config["train"]["batch_size"])

        obs_vec = []
        pi_vec = []
        z_vec = []

        for sample in samples:
            obs, pi, total_reward = sample
            obs_vec.append(obs)
            pi_vec.append(pi)
            if config["ranked_reward"]["update_rank"]:
                assert (
                    ranked_reward_buffer is not None
                ), "rank can only be updated when ranked reward is used"
                z_vec.append(ranked_reward_buffer.get_ranked_reward(total_reward))
            else:
                z_vec.append(total_reward)

        obs_vec = (
            np.array([x[0] for x in obs_vec], dtype=np.float32),
            np.array([x[1] for x in obs_vec], dtype=np.float32),
            np.array([x[2] for x in obs_vec], dtype=np.float32),
        )

        pi_vec = np.array(pi_vec, dtype=np.float32)
        z_vec = np.array(z_vec, dtype=np.float32)
        z_vec = np.expand_dims(z_vec, 1)

        obs_vec = tuple(
            torch.from_numpy(x).to(device).to(torch.float32) for x in obs_vec
        )
        pi_vec = torch.from_numpy(pi_vec).to(device)
        z_vec = torch.from_numpy(z_vec).to(device)

        pred_p, pred_v = model(*obs_vec)

        policy_loss = (
            torch.sum(-pi_vec * torch.log(pred_p + 1e-8))
            / pi_vec.shape[0]
            * config["train"]["policy_loss_factor"]
        )
        value_loss = mse_loss(pred_v, z_vec) * config["train"]["value_loss_factor"]

        loss = value_loss + policy_loss
        acc_loss_value.append(value_loss.cpu().detach().data.tolist())
        acc_loss_policy.append(policy_loss.cpu().detach().data.tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for i, loss in enumerate(acc_loss_value):
        logs.append(("value loss", loss, i + offset))

    for i, loss in enumerate(acc_loss_policy):
        logs.append(("policy loss", loss, i + offset))

    return logs
