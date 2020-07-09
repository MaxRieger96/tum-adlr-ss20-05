import os
from os import path

import numpy as np
import torch
from elevator_rl.yparams import YParams
from torch.nn.functional import mse_loss
from torch.optim import Adam

from elevator_rl.alphazero.model import NNModel
from elevator_rl.alphazero.ranked_reward import RankedRewardBuffer
from elevator_rl.alphazero.replay_buffer import ReplayBuffer
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.environment.elevator import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.example_houses import get_simple_house
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from datetime import datetime

config_name = os.environ["CONFIG_NAME"] if "CONFIG_NAME" in os.environ else "default"
yparams = YParams("config.yaml", config_name)
config = yparams.hparams
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{config_name}'
batch_count = (
        config["train"]["samples_per_iteration"] // config["train"]["batch_size"]
)


def train(
        model: NNModel,
        replay_buffer: ReplayBuffer,
        ranked_reward_buffer: RankedRewardBuffer,
        offset: int
):
    optimizer = Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
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


def write_hparams(writer: SummaryWriter):
    exp, ssi, sei = hparams(yparams.flatten(yparams.hparams), {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def main():
    writer = SummaryWriter(path.join(config["path"], run_name))
    write_hparams(writer=writer)

    house = get_simple_house()

    env = ElevatorEnv(house)
    env.render(method="matplotlib")

    replay_buffer = ReplayBuffer(capacity=config["replay_buffer"]["size"])
    ranked_reward_buffer = RankedRewardBuffer(
        capacity=config["ranked_reward"]["size"],
        threshold=config["ranked_reward"]["threshold"],
    )
    # TODO use parallel factory
    generator = Generator(env, ranked_reward_buffer=None)  # TODO make optional
    model = NNModel(
        house_observation_dims=env.get_observation().as_array()[0].shape[0],
        elevator_observation_dims=env.get_observation().as_array()[1].shape[0],
        policy_dims=ElevatorActionEnum.count(),
    )
    iteration_start = 0
    for i in range(iteration_start, config["train"]["iterations"]):
        print(f"iteration {i}: sampling started")
        for _ in range(config["train"]["episodes"]):
            observations, pis, total_reward, summary = generator.perform_episode(
                mcts_samples=config["mcts"]["samples"],
                mcts_temp=config["mcts"]["temp"],
                mcts_cpuct=config["mcts"]["cpuct"],
                mcts_observation_weight=config["mcts"]["observation_weight"],
                model=model,
            )
            for j, pi in enumerate(pis):
                sample = (observations[j], pi, total_reward)
                replay_buffer.push(sample)

        # TRAIN model
        logs = train(model, replay_buffer, ranked_reward_buffer, i * batch_count)
        for log in logs:
            writer.add_scalar(*log)
            print(log)


if __name__ == "__main__":
    main()
