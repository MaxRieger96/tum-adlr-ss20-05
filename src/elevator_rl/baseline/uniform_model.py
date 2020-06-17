import logging

import numpy as np

from elevator_rl.alphazero.model import Model
from elevator_rl.alphazero.sample_generator import EpisodeFactory
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.episode_summary import combine_summaries
from elevator_rl.environment.example_houses import get_simple_house

MCTS_SAMPLES = 200
MCTS_TEMP = 1
MCTS_CPUCT = 4
MCTS_OBSERVATION_WEIGHT = 1.0  # TODO change for modified mcts
EPISODES = 100
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
    env.render(method="matplotlib", step=0)
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
    print(combine_summaries(summaries))
    print(f"{MCTS_SAMPLES} mcts samples")


if __name__ == "__main__":
    main()
