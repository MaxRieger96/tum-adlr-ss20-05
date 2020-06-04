from enum import Enum
from typing import List

import numpy as np

from elevator_rl.elevator_env_render import render
from elevator_rl.example_houses import get_10_story_house
from elevator_rl.house import House


class ElevatorActionEnum(Enum):
    DOWN = -1
    OPEN = 0
    UP = 1
    IDLE = 2

    @staticmethod
    def count() -> int:
        return len([d for d in ElevatorActionEnum])


class ElevatorEnvAction:
    def __init__(self, elevator_idx: int, elevator_action: ElevatorActionEnum):
        self.elevator_idx: int = elevator_idx
        self.elevator_action: ElevatorActionEnum = elevator_action


class ElevatorEnv:
    # TODO env should give rewards and store waiting times for total rewards
    # TODO there should be observations
    def __init__(self, house: House):
        self.house: House = house
        self.next_elevator: int = 0
        self.transported_passenger_times: List[float] = []

    def step(self, env_action: ElevatorEnvAction) -> float:
        # move only if valid move
        if env_action.elevator_idx not in range(0, len(self.house.elevators)):
            raise ValueError("Elevator-ID does not exist")
        assert (
            env_action.elevator_idx == self.next_elevator
        ), "elevators should be controlled in the right order"

        start_time = self.house.time

        elevator = self.house.elevators[env_action.elevator_idx]

        # perform action
        if env_action.elevator_action == ElevatorActionEnum.DOWN:
            elevator.down()
        elif env_action.elevator_action == ElevatorActionEnum.UP:
            elevator.up()
        elif env_action.elevator_action == ElevatorActionEnum.OPEN:
            leaving_passengers = self.house.open_elevator(env_action.elevator_idx)
            # store waiting time
            self.transported_passenger_times += [
                elevator.time - p.waiting_since for p in leaving_passengers
            ]
        elif env_action.elevator_action == ElevatorActionEnum.IDLE:
            self.house.elevators[env_action.elevator_idx].idle()

        # find elevator which performs action next
        self.next_elevator = self.house.next_to_move()

        # time to elapse to
        new_time = self.house.elevators[self.next_elevator].time

        # run time until action takes place
        self.house.elapse_time_to(new_time)

        return -1 * self._quadratic_waiting_time(start_time, new_time)

    def render(self):
        render(self.house)

    def get_total_waiting_time(self) -> float:
        """
        gets the sum of all waiting times of passengers, this includes:
        - all passengers, which have be transported before
        - all passengers in elevators
        - randomly sampled passengers at floors with request signals
        :return: total_waiting_time: float
        """
        return sum(self._get_all_waiting_times())

    def _get_all_waiting_times(self) -> List[float]:
        """
        gets all waiting times of passengers, this includes:
        - all passengers, which have be transported before
        - all passengers in elevators
        - randomly sampled passengers at floors with request signals
        :return: waiting_times: List[float]
        """
        return (
            self.transported_passenger_times
            + self.house.get_waiting_time_for_all_waiting_passengers()
        )

    def _calc_passenger_waiting_time(
        self, start_time: float, until_time: float
    ) -> float:
        """
        calc the expected accumulated waiting time of passengers between start_time and
        until_time
        :return: accumulated_waiting_time: float
        """
        # use expected values of counts of passengers to calculate all waiting times
        time_delta = until_time - start_time
        passenger_count = self.house.get_expected_passenger_count(until_time)
        return time_delta * passenger_count

    def _quadratic_waiting_time(self, start_time: float, new_time: float) -> float:
        """
        computes the delta between quadratic waiting times from start_time to new_time
        :return:
        """
        arrival_times = self.house.get_arrival_time_for_all_waiting_passengers()
        if len(arrival_times) == 0:
            return 0
        arrival_times = np.array(arrival_times)
        waiting_times_at_start = np.maximum(0, start_time - arrival_times)
        waiting_times_at_new_time = new_time - arrival_times
        quadratic_deltas = waiting_times_at_new_time ** 2 - waiting_times_at_start ** 2
        return float(np.sum(quadratic_deltas))


def main():
    house = get_10_story_house()

    env = ElevatorEnv(house)
    for _ in range(100):
        for _ in range(9):
            for _ in range(4):
                action = ElevatorEnvAction(env.next_elevator, ElevatorActionEnum.UP)
                env.step(action)
        for _ in range(9):
            for _ in range(4):
                action = ElevatorEnvAction(env.next_elevator, ElevatorActionEnum.DOWN)
                env.step(action)

    print(house)


if __name__ == "__main__":
    main()
