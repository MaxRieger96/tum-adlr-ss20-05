from elevator_rl.alphazero.replay_buffer import ReplayBuffer
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.example_houses import get_simple_house

MCTS_SAMPLES = 20
MCTS_TEMP = 1
MCTS_CPUCT = 4
MCTS_OBSERVATION_WEIGHT = 0.0  # TODO change for modified mcts
REPLAY_BUFFER_SIZE = 3000
ITERATIONS = 10


def main():
    house = get_simple_house()

    env = ElevatorEnv(house)
    env.render(method="matplotlib", step=0)

    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)  # TODO use configs
    generator = Generator(env, ranked_reward_buffer=None)  # TODO use configs

    iteration_start = 0
    for i in range(iteration_start, ITERATIONS):
        print(f"iteration {i}: sampling started")
        observations, pis, total_reward = generator.perform_episode(
            mcts_samples=MCTS_SAMPLES,
            mcts_temp=MCTS_TEMP,
            mcts_cpuct=MCTS_CPUCT,
            mcts_observation_weight=MCTS_OBSERVATION_WEIGHT,
        )
        for i, pi in enumerate(pis):
            sample = (observations[i], pi, total_reward)
            replay_buffer.push(sample)
        # TRAIN model


if __name__ == "__main__":
    main()
