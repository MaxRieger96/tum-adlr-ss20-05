from __future__ import annotations

from enum import Enum
from typing import List
from typing import Set
from typing import TYPE_CHECKING

import numpy as np

from elevator_rl.environment.passenger import Passenger

if TYPE_CHECKING:
    from elevator_rl.environment.house import House

IDLE_TIME = 10.0
# TODO decide whether we should use the same time for all actions
#  MOVE_TIME = 2.0
#  ENTER_TIME = 1.0
#  DOOR_OPEN_TIME = 1.0
ANY_MOVE_TIME = 2.0


class ElevatorActionEnum(Enum):
    DOWN = 0
    OPEN = 1
    UP = 2
    # IDLE = 3  # TODO check if we can simply omit this

    @staticmethod
    def count() -> int:
        return len([d for d in ElevatorActionEnum])


class ElevatorEnvAction:
    def __init__(self, elevator_idx: int, elevator_action: ElevatorActionEnum):
        self.elevator_idx: int = elevator_idx
        self.elevator_action: ElevatorActionEnum = elevator_action


class Elevator:
    def __init__(
        self,
        capacity: int,
        floor: int,
        passengers: Set[Passenger],
        floor_requests: np.ndarray,
        house: "House",
    ):
        self.capacity: int = capacity
        self.floor: int = floor
        assert len(passengers) <= capacity
        self.passengers: Set[Passenger] = passengers
        assert floor_requests.dtype == bool
        self.floor_requests: np.ndarray = floor_requests
        self.time: float = 0
        self.house: "House" = house

    def free_places(self) -> int:
        return self.capacity - len(self.passengers)

    def up(self):
        assert self.floor + 1 < self.house.number_of_floors
        self.floor += 1
        self.time += ANY_MOVE_TIME  # FIXME

    def down(self):
        assert self.floor > 0
        self.floor -= 1
        self.time += ANY_MOVE_TIME  # FIXME

    def passengers_enter_elevator(
        self, entering_passengers: Set[Passenger], new_floor_requests: List[int],
    ):
        assert len(self.passengers) <= self.capacity
        self.passengers = self.passengers | entering_passengers
        assert len(self.passengers) <= self.capacity
        self.floor_requests[self.floor] = False
        self.floor_requests[new_floor_requests] = True
        # TODO set better estimates of time taken for entering and leaving
        # self.time += DOOR_OPEN_TIME + len(entering_passengers) * ENTER_TIME
        self.time += ANY_MOVE_TIME / 2  # FIXME

    def passengers_exit_elevator(self, leaving_passengers: Set[Passenger]):
        self.passengers = self.passengers - leaving_passengers
        # self.time += DOOR_OPEN_TIME + len(leaving_passengers) * ENTER_TIME
        self.time += ANY_MOVE_TIME / 2  # FIXME

    def idle(self):
        self.time += IDLE_TIME

    def valid_actions(self) -> np.ndarray:
        valid_actions = np.array([1 for _ in ElevatorActionEnum])
        if self.floor == 0:
            valid_actions[ElevatorActionEnum.DOWN.value] = 0
        if self.floor + 1 == self.house.number_of_floors:
            valid_actions[ElevatorActionEnum.UP.value] = 0
        return valid_actions
