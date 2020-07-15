import os
from copy import deepcopy
from datetime import datetime
from os import path
from typing import Dict
from typing import List

import numpy as np
import torch
from torch.multiprocessing import Process
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from elevator_rl.alphazero.model import NNModel
from elevator_rl.alphazero.ranked_reward import RankedRewardBuffer
from elevator_rl.alphazero.replay_buffer import ReplayBuffer
from elevator_rl.alphazero.sample_generator import EpisodeFactory
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.alphazero.sample_generator import SingleProcessEpisodeFactory
from elevator_rl.environment.elevator import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.episode_summary import accumulate_summaries
from elevator_rl.environment.episode_summary import Summary
from elevator_rl.environment.example_houses import produce_house
from elevator_rl.yparams import YParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def write_hparams(writer: SummaryWriter, yparams: YParams):
    exp, ssi, sei = hparams(yparams.flatten(yparams.hparams), {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def write_episode_summary(
    writer: SummaryWriter, summary: Summary, index: int, name: str
):
    name = name + "_"
    writer.add_scalar(
        name + "quadratic_waiting_time", summary.quadratic_waiting_time, index
    )
    writer.add_scalar(name + "waiting_time", summary.waiting_time, index)
    writer.add_scalar(
        name + "percent_transported", summary.percent_transported(), index
    )
    writer.add_scalar(
        name + "avg_waiting_time_transported",
        summary.avg_waiting_time_transported,
        index,
    )
    writer.add_scalar(
        name + "avg_waiting_time_per_person",
        summary.avg_waiting_time_per_person,
        index,
    )
    writer.add_scalar(name + "nr_waiting", summary.nr_passengers_waiting, index)
    writer.add_scalar(name + "nr_transported", summary.nr_passengers_transported, index)


def write_episode_summaries(
    writer: SummaryWriter, summaries: List[Summary], index: int
):
    # TODO make this nice in tensorboard using custom scalars
    #  https://stackoverflow.com/questions/37146614/tensorboard-plot-training-and-validation-losses-on-the-same-graph
    for name, accumulator in {
        "avg": lambda x: sum(x) / len(x),
        "max": lambda x: max(x),
        "min": lambda x: min(x),
    }.items():
        write_episode_summary(
            writer, accumulate_summaries(summaries, accumulator), index, name
        )


def main(config_name: str):
    yparams = YParams("config.yaml", config_name)
    config = yparams.hparams
    run_name = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{config_name}'

    writer = SummaryWriter(path.join(config["path"], run_name))
    write_hparams(writer=writer, yparams=yparams)

    batch_count = (
        config["train"]["samples_per_iteration"] // config["train"]["batch_size"]
    )
    house = produce_house(
        elevator_capacity=config["house"]["elevator_capacity"],
        number_of_elevators=config["house"]["number_of_elevators"],
        number_of_floors=config["house"]["number_of_floors"],
    )

    env = ElevatorEnv(house)
    env.render(method="matplotlib")

    replay_buffer = ReplayBuffer(capacity=config["replay_buffer"]["size"])
    ranked_reward_buffer = RankedRewardBuffer(
        capacity=config["ranked_reward"]["size"],
        threshold=config["ranked_reward"]["threshold"],
    )

    generator = Generator(env, ranked_reward_buffer)
    if config["train"]["n_processes"] > 1:
        factory = EpisodeFactory(generator)
    else:
        factory = SingleProcessEpisodeFactory(generator)

    model = NNModel(
        house_observation_dims=env.get_observation().as_array()[0].shape[0],
        elevator_observation_dims=env.get_observation().as_array()[1].shape[0],
        policy_dims=ElevatorActionEnum.count(),
    )

    iteration_start = 0
    for i in range(iteration_start, config["train"]["iterations"]):

        print(f"\niteration {i}: sampling started")

        episodes = factory.create_episodes(
            n_episodes=config["train"]["episodes"],
            n_processes=config["train"]["n_processes"],
            mcts_samples=config["mcts"]["samples"],
            mcts_temp=config["mcts"]["temp"],
            mcts_cpuct=config["mcts"]["cpuct"],
            mcts_observation_weight=config["mcts"]["observation_weight"],
            model=model,
        )

        summaries = []
        for episode_index, e in enumerate(episodes):
            assert len(episodes) < batch_count, "the tensorboard indices make no sense"
            observations, pis, total_reward, summary = e
            for j, pi in enumerate(pis):
                sample = (observations[j], pi, total_reward)
                replay_buffer.push(sample)
            summaries.append(summary)

        write_episode_summaries(writer, summaries, i * batch_count)

        if i > 0 and i % 3 == 0 and config["visualize_iterations"]:
            # Visualization Process outputting a video for each iteration
            p = Process(
                target=generator.perform_episode,
                args=(
                    config["mcts"]["samples"],
                    config["mcts"]["temp"],
                    config["mcts"]["cpuct"],
                    config["mcts"]["observation_weight"],
                    deepcopy(model),
                    True,
                    i,
                    run_name,
                ),
            )
            p.start()

        # TRAIN model
        logs = train(
            model, replay_buffer, ranked_reward_buffer, i * batch_count, config
        )
        for log in logs:
            writer.add_scalar(*log)


if __name__ == "__main__":
    main(
        config_name=os.environ["CONFIG_NAME"]
        if "CONFIG_NAME" in os.environ
        else "default"
    )
