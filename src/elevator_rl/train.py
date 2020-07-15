import os
from copy import deepcopy
from datetime import datetime
from os import path
from typing import Dict

import numpy as np
import torch
from torch.multiprocessing import Process
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from elevator_rl.alphazero.model import Model
from elevator_rl.alphazero.model import NNModel
from elevator_rl.alphazero.ranked_reward import RankedRewardBuffer
from elevator_rl.alphazero.replay_buffer import ReplayBuffer
from elevator_rl.alphazero.sample_generator import EpisodeFactory
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.alphazero.sample_generator import MultiProcessEpisodeFactory
from elevator_rl.alphazero.sample_generator import SingleProcessEpisodeFactory
from elevator_rl.alphazero.tensorboard import Logger
from elevator_rl.environment.elevator import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.example_houses import produce_house
from elevator_rl.yparams import YParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_video_images(
    generator: Generator, config: Dict, model: Model, i: int, run_name: str
):
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


def main(config_name: str):
    yparams = YParams("config.yaml", config_name)
    config = yparams.hparams
    run_name = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{config_name}'
    assert not (
        config["offline_training"] and not config["pretrained_path"]
    ), "Offline training requires pretrained buffer"

    logger = Logger(SummaryWriter(path.join(config["path"], run_name)))
    logger.write_hparams(yparams=yparams)

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
    factory: EpisodeFactory
    if config["train"]["n_processes"] > 1:
        factory = MultiProcessEpisodeFactory(generator)
    else:
        factory = SingleProcessEpisodeFactory(generator)

    model = NNModel(
        house_observation_dims=env.get_observation().as_array()[0].shape[0],
        elevator_observation_dims=env.get_observation().as_array()[1].shape[0],
        policy_dims=ElevatorActionEnum.count(),
    )

    if config["pretrained_path"]:
        checkpoint = torch.load(config["pretrained_path"])
        env = checkpoint["environment"]
        model.load_state_dict(checkpoint["model_state_dict"])
        iteration_start = checkpoint["iteration_start"]
        replay_buffer = checkpoint["replay_buffer"]
        ranked_reward_buffer = checkpoint["ranked_reward_buffer"]
    else:
        iteration_start = 0

    for i in range(iteration_start, config["train"]["iterations"]):
        print(f"\niteration {i}")
        if not config["offline_training"]:
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
                observations, pis, total_reward, summary = e
                for j, pi in enumerate(pis):
                    sample = (observations[j], pi, total_reward)
                    replay_buffer.push(sample)
                summaries.append(summary)

            logger.write_episode_summaries(summaries, i * batch_count)

            if i > 0 and i % 3 == 0 and config["visualize_iterations"]:
                create_video_images(
                    generator=generator,
                    config=config,
                    model=model,
                    i=i,
                    run_name=run_name,
                )

        # TRAIN model
        logs = train(
            model, replay_buffer, ranked_reward_buffer, i * batch_count, config
        )
        logger.log_train(logs)

        if config["save_iterations"]:
            torch.save(
                {
                    "environment": env,
                    "iteration_start": i + 1,
                    "model_state_dict": model.state_dict(),
                    "replay_buffer": replay_buffer,
                    "ranked_reward_buffer": ranked_reward_buffer,
                },
                path.join(config["path"], run_name, f"model_save_{i}.pth"),
            )


if __name__ == "__main__":
    loaded_config = (
        os.environ["CONFIG_NAME"] if "CONFIG_NAME" in os.environ else "default"
    )
    print(loaded_config)
    main(config_name=loaded_config)
