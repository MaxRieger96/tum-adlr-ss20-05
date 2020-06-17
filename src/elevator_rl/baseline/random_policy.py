import numpy as np
import torch
from torch.distributions import Categorical

from elevator_rl.environment.elevator import ElevatorEnvAction
from elevator_rl.environment.elevator_env import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.episode_summary import combine_summaries
from elevator_rl.environment.example_houses import get_5_story_house


class RandomPolicy:
    def get_action(self, env: ElevatorEnv):
        policy = np.array([1 / ElevatorActionEnum.count() for _ in ElevatorActionEnum])
        valid_actions = env.house.elevators[env.next_elevator].valid_actions()
        policy = policy * valid_actions  # masking invalid moves
        policy /= np.sum(policy)  # renormalize

        # sample action
        cat = Categorical(torch.from_numpy(policy))
        action = cat.sample()
        return action.numpy().flatten()[0]


def main():
    summaries = []
    for i in range(100):
        print(i)
        house = get_5_story_house()

        env = ElevatorEnv(house)
        # env.render()
        random_policy = RandomPolicy()
        step = 0
        while not env.is_end_of_day():
            random_action = random_policy.get_action(env)
            env.step(
                ElevatorEnvAction(env.next_elevator, ElevatorActionEnum(random_action))
            )
            step += 1
            # env.render()
            # env.render(method="file", step=step)
        # print("Total reward at the end of day: {}".format(env.reward_acc))
        print(env.get_summary())
        summaries.append(env.get_summary())

    print(combine_summaries(summaries))


if __name__ == "__main__":
    main()
