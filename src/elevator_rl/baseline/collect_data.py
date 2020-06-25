import logging
import pickle
from datetime import datetime

from elevator_rl.alphazero.sample_generator import EpisodeFactory
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.baseline.random_policy import RandomPolicy
from elevator_rl.baseline.uniform_model import UniformModel
from elevator_rl.environment.elevator import ElevatorActionEnum
from elevator_rl.environment.elevator import ElevatorEnvAction
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.episode_summary import combine_summaries
from elevator_rl.environment.example_houses import produce_house


MCTS_TEMP = 0.01
MCTS_CPUCT = 4
MCTS_OBSERVATION_WEIGHT = 1.0
EPISODES = 32
PROCESSES = 16
LOG_FILE = f'data_collection_log_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'


def print_double(s: str):
    print(s)

    with open(LOG_FILE, "a") as myfile:
        myfile.write(s + "\n")


def main():
    for floors in range(3, 11):
        print_double(f"test houses with {floors} floors")
        for elevators in [1, 2]:
            print_double(f"test houses with {elevators} elevators")
            create_random(elevators, floors)
            for MCTS_SAMPLES in [10, 20, 50, 100, 200]:
                print_double(f"use {MCTS_SAMPLES} mcts samples")
                logging.basicConfig(level=logging.ERROR)
                house = produce_house(20, elevators, floors)

                env = ElevatorEnv(house)
                # env.render(method="matplotlib", step=0)
                generator = Generator(env, ranked_reward_buffer=None)

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

                print_double("")
                with open(
                    f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
                    f"_mcts{MCTS_SAMPLES}"
                    f"_floors{env.house.number_of_floors}"
                    f"_elevs{len(env.house.elevators)}",
                    "wb",
                ) as handle:
                    pickle.dump(summaries, handle, protocol=pickle.HIGHEST_PROTOCOL)
                avg, stddev = combine_summaries(summaries)
                print_double(str(avg))
                print_double(str(stddev))
                print_double(
                    f"{MCTS_SAMPLES} mcts samples, mcts_temp: {MCTS_TEMP}, "
                    f"floors{env.house.number_of_floors}, elevs{len(env.house.elevators)}"
                )
                print_double("\n\n")


def create_random(elevators, floors):
    summaries = []
    house = produce_house(20, elevators, floors)
    env = ElevatorEnv(house)
    for i in range(100):
        house = produce_house(20, elevators, floors)
        env = ElevatorEnv(house)
        random_policy = RandomPolicy()
        step = 0
        while not env.is_end_of_day():
            random_action = random_policy.get_action(env)
            env.step(
                ElevatorEnvAction(env.next_elevator, ElevatorActionEnum(random_action))
            )
            step += 1
        summaries.append(env.get_summary())
    with open(
        f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
        f"_mcts{0}"
        f"_floors{env.house.number_of_floors}"
        f"_elevs{len(env.house.elevators)}",
        "wb",
    ) as handle:
        pickle.dump(summaries, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
