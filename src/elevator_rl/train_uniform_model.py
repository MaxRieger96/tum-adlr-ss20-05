import logging

from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.example_houses import get_simple_house

MCTS_SAMPLES = 20
MCTS_TEMP = 1
MCTS_CPUCT = 4
MCTS_OBSERVATION_WEIGHT = 0.0  # TODO change for modified mcts
REPLAY_BUFFER_SIZE = 3000
ITERATIONS = 2


def main():
    logging.basicConfig(level=logging.ERROR)
    house = get_simple_house()

    env = ElevatorEnv(house)
    env.render(method="matplotlib", step=0)
    generator = Generator(env, ranked_reward_buffer=None)  # TODO use configs

    iteration_start = 0
    for i in range(iteration_start, ITERATIONS):
        print(f"iteration {i}: sampling started")
        observations, pis, total_reward = generator.perform_episode(
            mcts_samples=MCTS_SAMPLES,
            mcts_temp=MCTS_TEMP,
            mcts_cpuct=MCTS_CPUCT,
            mcts_observation_weight=MCTS_OBSERVATION_WEIGHT,
            no_nn=True,  # THIS SETS MCTS TO THE UNIFORM MODEL
        )
        print("Total reward at the end of the day: {}".format(total_reward))


if __name__ == "__main__":
    main()
