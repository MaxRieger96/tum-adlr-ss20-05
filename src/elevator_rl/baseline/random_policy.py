import numpy as np
import os
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from elevator_rl.alphazero.tensorboard import Logger
from elevator_rl.environment.elevator import ElevatorEnvAction
from elevator_rl.environment.elevator_env import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from pathlib import Path
from elevator_rl.environment.example_houses import get_10_story_single_elev_house, get_simple_house
from elevator_rl.yparams import YParams


class RandomPolicy:
    def eval(self):
        """
        this is only here, because we use it for NNModels
        :return:
        """
        pass

    def get_action(self, env: ElevatorEnv):
        policy = np.array([1 / ElevatorActionEnum.count() for _ in ElevatorActionEnum])
        valid_actions = env.house.elevators[env.next_elevator].valid_actions()
        policy = policy * valid_actions  # masking invalid moves
        policy /= np.sum(policy)  # renormalize

        # sample action
        cat = Categorical(torch.from_numpy(policy))
        action = cat.sample()
        return action.numpy().flatten()[0]


def main(render: bool):
    from os import path
    run_name = "simple_house_random_policy"
    logger = Logger(SummaryWriter(path.join("../../../runs", run_name)))
    yparams = YParams("../config.yaml", "default")
    config = yparams.hparams
    batch_count = (
            config["train"]["samples_per_iteration"] // config["train"]["batch_size"]
    )
    for i in range(config["train"]["iterations"]):
        summaries = []
        for episode in range(config["train"]["episodes"]):
            print(i)
            house = get_simple_house()

            env = ElevatorEnv(house)
            random_policy = RandomPolicy()
            step = 0
            while not env.is_end_of_day():
                random_action = random_policy.get_action(env)
                prev_time = env.house.time
                action = ElevatorEnvAction(
                    env.next_elevator, ElevatorActionEnum(random_action)
                )
                env.step(action)
                step += 1
                if render:
                    root_dir = os.path.dirname(os.path.abspath(__file__))
                    path = os.path.join(
                        root_dir,
                        "{}/../plots/run_{}/iteration{}".format(
                            root_dir, run_name, i
                        ),
                    )
                    Path(path).mkdir(parents=True, exist_ok=True)
                    env.render(method="file", path=path, prev_time=prev_time, action=action)

            # print("Total reward at the end of day: {}".format(env.reward_acc))
            print(env.get_summary())
            summaries.append(env.get_summary())
        logger.write_episode_summaries(summaries, i * batch_count)
    # avg, stddev = combine_summaries(summaries)
    # print(avg)
    # print(stddev)


if __name__ == "__main__":
    main(render=False)
