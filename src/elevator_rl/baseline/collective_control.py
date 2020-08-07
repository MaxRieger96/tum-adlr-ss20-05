import os
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from elevator_rl.alphazero.tensorboard import Logger
from elevator_rl.environment.elevator import ElevatorEnvAction
from elevator_rl.environment.elevator_env import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.example_houses import get_simple_house
from elevator_rl.yparams import YParams


class CollectiveControl:
    def __init__(self):
        self.elevators_latest_direction = {}

    def latest_direction_is(self, env: ElevatorEnv, direction: ElevatorActionEnum):
        return (
            env.next_elevator in self.elevators_latest_direction
            and self.elevators_latest_direction[env.next_elevator] == direction
        )

    def get_action(self, env: ElevatorEnv):
        elevator = env.house.elevators[env.next_elevator]

        if elevator.floor == 0:
            self.elevators_latest_direction[env.next_elevator] = ElevatorActionEnum.UP
        elif elevator.floor == env.house.number_of_floors - 1:
            self.elevators_latest_direction[env.next_elevator] = ElevatorActionEnum.DOWN

        # collect an answer all calls in one direction, then reverse and repeat
        if elevator.floor == env.house.number_of_floors - 1:
            floor_requests_above = []
        else:
            floor_requests_above = np.any(elevator.floor_requests[elevator.floor + 1 :])
        floor_requests_below = np.any(elevator.floor_requests[: elevator.floor])
        floor_requests_current = elevator.floor_requests[elevator.floor]

        if elevator.floor == env.house.number_of_floors - 1:
            house_requests_above = []
        else:
            house_requests_above = np.any(
                env.house.up_requests[elevator.floor + 1 :]
            ) or np.any(env.house.down_requests[elevator.floor + 1 :])
        house_requests_below = np.any(
            env.house.up_requests[: elevator.floor]
        ) or np.any(env.house.down_requests[: elevator.floor])
        if floor_requests_current:  # Passengers want to leave
            return ElevatorActionEnum.OPEN
        elif env.house.up_requests[elevator.floor] and self.latest_direction_is(
            env, ElevatorActionEnum.UP
        ):  # Pick up on way up
            return ElevatorActionEnum.OPEN
        elif env.house.down_requests[elevator.floor] and self.latest_direction_is(
            env, ElevatorActionEnum.DOWN
        ):  # Pick up on way down
            return ElevatorActionEnum.OPEN
        elif self.latest_direction_is(env, ElevatorActionEnum.UP) and (
            floor_requests_above or house_requests_above
        ):  # More to do at the top
            self.elevators_latest_direction[env.next_elevator] = ElevatorActionEnum.UP
            return ElevatorActionEnum.UP
        elif self.latest_direction_is(env, ElevatorActionEnum.DOWN) and (
            floor_requests_below or house_requests_below
        ):  # More to do at the bottom
            self.elevators_latest_direction[env.next_elevator] = ElevatorActionEnum.DOWN
            return ElevatorActionEnum.DOWN
        # COLD START SCENARIOS
        elif (
            floor_requests_below
        ):  # first person gets in at the floor the elevator already is at
            self.elevators_latest_direction[env.next_elevator] = ElevatorActionEnum.DOWN
            return ElevatorActionEnum.DOWN
        elif (
            floor_requests_above
        ):  # first person gets in at the floor the elevator already is at
            self.elevators_latest_direction[env.next_elevator] = ElevatorActionEnum.UP
            return ElevatorActionEnum.UP
        elif house_requests_below:
            self.elevators_latest_direction[env.next_elevator] = ElevatorActionEnum.DOWN
            return ElevatorActionEnum.DOWN
        elif house_requests_above:
            self.elevators_latest_direction[env.next_elevator] = ElevatorActionEnum.UP
            return ElevatorActionEnum.UP
        else:
            return ElevatorActionEnum.OPEN


def main(render: bool):
    from os import path

    run_name = "simple_house_cc"
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
            collective_control = CollectiveControl()
            step = 0
            while not env.is_end_of_day():
                cc_action = collective_control.get_action(env)
                prev_time = env.house.time
                action = ElevatorEnvAction(
                    env.next_elevator, ElevatorActionEnum(cc_action)
                )
                env.step(action)
                step += 1
                if render:
                    root_dir = os.path.dirname(os.path.abspath(__file__))
                    path = os.path.join(
                        root_dir,
                        "{}/../plots/run_{}/iteration{}".format(root_dir, run_name, i),
                    )
                    Path(path).mkdir(parents=True, exist_ok=True)
                    env.render(
                        method="file", path=path, prev_time=prev_time, action=action
                    )

            # print("Total reward at the end of day: {}".format(env.reward_acc))
            print(env.get_summary())
            summaries.append(env.get_summary())
        logger.write_episode_summaries(summaries, i * batch_count)
    # avg, stddev = combine_summaries(summaries)
    # print(avg)
    # print(stddev)


if __name__ == "__main__":
    main(render=False)
