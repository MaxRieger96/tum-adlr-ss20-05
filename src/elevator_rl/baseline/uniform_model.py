import logging
import pickle
from datetime import datetime

import numpy as np

from elevator_rl.alphazero.model import Model
from elevator_rl.alphazero.sample_generator import EpisodeFactory
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.episode_summary import combine_summaries
from elevator_rl.environment.example_houses import get_simple_house

MCTS_SAMPLES = 10
MCTS_TEMP = 0.01
MCTS_CPUCT = 4
MCTS_OBSERVATION_WEIGHT = 1.0  # TODO change for modified mcts
EPISODES = 16
PROCESSES = 16


class UniformModel(Model):
    def eval(self):
        """
        this is only here, because we use it for NNModels
        :return:
        """
        pass

    def get_policy_and_value(self, env: ElevatorEnv):
        value = 0
        valid_actions = env.house.elevators[env.next_elevator].valid_actions()
        policy = valid_actions / np.sum(valid_actions)  # normalize
        return policy, value


def main():

    logging.basicConfig(level=logging.ERROR)
    house = get_simple_house()

    env = ElevatorEnv(house)
    env.render(method="matplotlib")
    generator = Generator(env, ranked_reward_buffer=None)  # TODO use configs

    factory = EpisodeFactory(generator)
    model = UniformModel()
    episodes = factory.create_episodes(
        EPISODES,
        PROCESSES,
        MCTS_SAMPLES,
        MCTS_TEMP,
        MCTS_CPUCT,
        MCTS_OBSERVATION_WEIGHT,
        model,
    )
    summaries = [e[3] for e in episodes]

    print()
    with open(
        f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
        f"_mcts{MCTS_SAMPLES}"
        f"_floors{env.house.number_of_floors}"
        f"_elevs{len(env.house.elevators)}",
        "wb",
    ) as handle:
        pickle.dump(summaries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    avg, stddev = combine_summaries(summaries)
    print(avg)
    print(stddev)
    print(
        f"{MCTS_SAMPLES} mcts samples, mcts_temp: {MCTS_TEMP}, "
        f"floors{env.house.number_of_floors}, elevs{len(env.house.elevators)}"
    )
    print("\n\n")


if __name__ == "__main__":
    main()
