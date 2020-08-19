from copy import deepcopy
from statistics import stdev
from typing import Dict
from typing import Optional

import torch

from elevator_rl.alphazero.model import NNModel
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.alphazero.sample_generator import MultiProcessEpisodeFactory
from elevator_rl.environment.elevator import ElevatorActionEnum
from elevator_rl.environment.episode_summary import accumulate_summaries
from elevator_rl.yparams import YParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluation(
    model_path: str, config: Dict, nr_episodes: int, mcts_temp: Optional[float] = None
):
    # load model
    checkpoint = torch.load(model_path)
    env = checkpoint["environment"]
    model = NNModel(
        house_observation_dims=env.get_observation().as_array()[0].shape[0],
        elevator_observation_dims=env.get_observation().as_array()[1].shape[0],
        policy_dims=ElevatorActionEnum.count(),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    generator = Generator(env, None)
    factory = MultiProcessEpisodeFactory(generator)

    # generate episodes
    if mcts_temp is None:
        mcts_temp = 0  # temp of 0 for "optimal episodes"
    episodes = factory.create_episodes(
        nr_episodes,
        config["train"]["n_processes"],
        config["mcts"]["samples"],
        mcts_temp,
        config["mcts"]["cpuct"],
        config["mcts"]["observation_weight"],
        deepcopy(model),
    )

    # collect summaries
    summaries = []
    for episode_index, e in enumerate(episodes):
        observations, pis, total_reward, summary = e
        summaries.append(summary)

    # compute statistics over summaries
    avg_summary = accumulate_summaries(summaries, lambda x: sum(x) / len(x))
    stdev_summary = abs(accumulate_summaries(summaries, lambda x: stdev(x)))
    # upper_stdev_summary = avg_summary + stdev_summary
    # lower_stdev_summary = avg_summary - stdev_summary

    print(f"\nEvaluation of {model_path}")
    print("Averages:")
    print(avg_summary)
    print("Std Deviation:")
    print(stdev_summary)


def run_multiple_evaluations(n_episodes: int):
    model_paths = [
        "/home/max/tum-adlr-ss20-05/runs/"
        "2020-08-17_16:50:54_optimized_params_penalty_trick/model_save_111.pth",
        "/home/max/tum-adlr-ss20-05/runs/"
        "2020-08-17_16:50:54_optimized_params_penalty_trick/model_save_111.pth",
        "/home/max/tum-adlr-ss20-05/runs/"
        "2020-08-19_10:38:06_2elev_3floor_continued/model_save_54.pth",
        "/home/max/tum-adlr-ss20-05/runs/"
        "2020-08-17_18:51:14_without_early_reward/model_save_77.pth",
    ]
    config_names = [
        "optimized_params",
        "optimized_params",
        "2elev_3floor",
        "without_early_reward",
    ]
    mcts_temps = [0, 1, 1, 1]
    for i, model_path in enumerate(model_paths):
        yparams = YParams("config.yaml", config_names[i])
        config = yparams.hparams
        run_evaluation(model_path, config, n_episodes, mcts_temps[i])


if __name__ == "__main__":
    run_multiple_evaluations(8)
