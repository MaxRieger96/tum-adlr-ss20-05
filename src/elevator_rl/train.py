import os
from datetime import datetime
from datetime import timedelta
from os import path
from typing import Dict

import torch
from torch.multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter

from elevator_rl.alphazero.model import NNModel
from elevator_rl.alphazero.model import train
from elevator_rl.alphazero.ranked_reward import RankedRewardBuffer
from elevator_rl.alphazero.replay_buffer import ReplayBuffer
from elevator_rl.alphazero.sample_generator import EpisodeFactory
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.alphazero.sample_generator import MultiProcessEpisodeFactory
from elevator_rl.alphazero.sample_generator import SingleProcessEpisodeFactory
from elevator_rl.alphazero.tensorboard import Logger
from elevator_rl.baseline.uniform_model import UniformModel
from elevator_rl.environment.elevator import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.episode_summary import accumulate_summaries
from elevator_rl.environment.example_houses import produce_house
from elevator_rl.evaluation_logging_process import evaluation_process
from elevator_rl.evaluation_logging_process import EvaluationLoggingProcess
from elevator_rl.yparams import YParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(
    env: ElevatorEnv,
    i: int,
    model: NNModel,
    replay_buffer: ReplayBuffer,
    ranked_reward_buffer: RankedRewardBuffer,
    config: Dict,
    run_name: str,
):
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


def learning_loop(
    config: Dict,
    run_name: str,
    yparams: YParams,
    time_out: timedelta = timedelta(hours=24),
):
    best_waiting_time = None

    start_time = datetime.now()

    logger = Logger(SummaryWriter(path.join(config["path"], run_name)))
    logger.write_hparams(yparams=yparams)
    eval_logging_process = EvaluationLoggingProcess(config, run_name)

    batch_count = (
        config["train"]["samples_per_iteration"] // config["train"]["batch_size"]
    )
    house = produce_house(
        elevator_capacity=config["house"]["elevator_capacity"],
        number_of_elevators=config["house"]["number_of_elevators"],
        number_of_floors=config["house"]["number_of_floors"],
    )

    env = ElevatorEnv(house)
    # env.render(method="matplotlib")

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

    if config["pure_mcts"]:
        model = UniformModel()
    else:
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
        # stop after timeout
        if datetime.now() - start_time > time_out:
            print(f"stopping because of timeout after {time_out}")
            break

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
            waiting_time = accumulate_summaries(
                summaries, lambda x: sum(x) / len(x)
            ).avg_waiting_time_per_person

            if best_waiting_time is None or waiting_time < best_waiting_time:
                best_waiting_time = waiting_time
                save_model(
                    env, i, model, replay_buffer, ranked_reward_buffer, config, run_name
                )

            if i > 0 and i % 3 == 0:
                logger.plot_summaries(False, i)

            if i > 0 and i % 10 == 0 and False:  # FIXME
                p = Process(
                    target=evaluation_process,
                    args=(generator, config, model, i, run_name, eval_logging_process),
                )
                p.start()

        # TRAIN model
        if not config["pure_mcts"]:
            logs = train(
                model, replay_buffer, ranked_reward_buffer, i * batch_count, config
            )
            logger.log_train(logs)

            if config["save_iterations"]:
                save_model(
                    env, i, model, replay_buffer, ranked_reward_buffer, config, run_name
                )


def main(config_name: str):
    yparams = YParams("config.yaml", config_name)
    config = yparams.hparams
    run_name = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{config_name}'
    assert not (
        config["offline_training"] and not config["pretrained_path"]
    ), "Offline training requires pretrained buffer"
    learning_loop(config, run_name, yparams)


if __name__ == "__main__":
    config_name = (
        os.environ["CONFIG_NAME"] if "CONFIG_NAME" in os.environ else "default"
    )
    loaded_config = config_name
    print(f"loaded config {config_name}")
    print(loaded_config)
    main(config_name=loaded_config)
