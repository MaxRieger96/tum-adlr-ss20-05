import logging

import numpy as np

from elevator_rl.alphazero.model import Model
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.example_houses import get_simple_house

MCTS_SAMPLES = 100
MCTS_TEMP = 1
MCTS_CPUCT = 4
MCTS_OBSERVATION_WEIGHT = 1.0  # TODO change for modified mcts
ITERATIONS = 1
EPISODES_PER_ITERATION = 1  # TODO move this to config


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

    iteration_start = 0
    for i in range(iteration_start, ITERATIONS):
        print(f"iteration {i}: sampling started")
        for _ in range(EPISODES_PER_ITERATION):
            observations, pis, total_reward, summary = generator.perform_episode(
                mcts_samples=MCTS_SAMPLES,
                mcts_temp=MCTS_TEMP,
                mcts_cpuct=MCTS_CPUCT,
                mcts_observation_weight=MCTS_OBSERVATION_WEIGHT,
                model=UniformModel(),
            )
            print()
            print(summary)


if __name__ == "__main__":
    main()
