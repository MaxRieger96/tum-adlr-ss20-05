from typing import List
from typing import Tuple

import numpy as np

from elevator_rl.environment.elevator import Elevator
from elevator_rl.environment.house import House

ObservationType = Tuple[np.ndarray, np.ndarray, List[np.ndarray]]


def to_one_hot(value: int, vector_length: int) -> List[bool]:
    return [i == value for i in range(vector_length)]


class Observation:
    def __init__(self, house: House, next_elevator_idx: int):
        self.down_requests: np.ndarray = house.down_requests.astype(np.float32)
        self.up_requests: np.ndarray = house.up_requests.astype(np.float32)

        # instead of waiting since, use waiting time
        self.down_requests_waiting_time: np.ndarray = (
            house.time - house.down_requests_waiting_since * house.down_requests
        ).astype(np.float32)
        self.up_requests_waiting_time: np.ndarray = (
            house.time - house.up_requests_waiting_since * house.up_requests
        ).astype(np.float32)

        # single vectors for all elevators
        self.other_elevators: List[np.ndarray] = []
        for elevator_idx, elevator in enumerate(house.elevators):
            if elevator_idx == next_elevator_idx:
                self.next_elevator = self.elevator_to_array(elevator)
            else:
                self.other_elevators.append(self.elevator_to_array(elevator))

    @staticmethod
    def elevator_to_array(elevator: Elevator) -> np.ndarray:
        return np.concatenate(
            [
                [elevator.free_places() * 1.0],
                np.array(
                    to_one_hot(elevator.floor, len(elevator.floor_requests)),
                    dtype=np.float32,
                ),
                elevator.floor_requests.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    def as_array(self) -> ObservationType:
        return (
            np.concatenate(
                [
                    self.down_requests,
                    self.down_requests_waiting_time,
                    self.up_requests,
                    self.up_requests_waiting_time,
                ],
                axis=0,
            ),
            self.next_elevator,
            [e for e in self.other_elevators],
        )
