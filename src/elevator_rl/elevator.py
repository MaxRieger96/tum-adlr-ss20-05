from __future__ import annotations

from typing import List
from typing import Set
from typing import TYPE_CHECKING

import numpy as np

from elevator_rl.passenger import Passenger

if TYPE_CHECKING:
    from elevator_rl.house import House

MOVE_TIME = 2.0  # TODO find a good time values
ENTER_TIME = 1.0
DOOR_OPEN_TIME = 1.0
IDLE_TIME = 10.0  # TODO decide if we want to have the option to idle


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
        self.time += MOVE_TIME

    def down(self):
        assert self.floor > 0
        self.floor -= 1
        self.time += MOVE_TIME

    def passengers_enter_elevator(
        self, entering_passengers: Set[Passenger], new_floor_requests: List[int],
    ):
        assert len(self.passengers) <= self.capacity
        self.passengers = self.passengers | entering_passengers
        self.floor_requests[self.floor] = False
        self.floor_requests[new_floor_requests] = True
        # TODO set better estimates of time taken for entering and leaving
        self.time += DOOR_OPEN_TIME + len(entering_passengers) * ENTER_TIME

    def passengers_exit_elevator(self, leaving_passengers: Set[Passenger]):
        self.passengers = self.passengers - leaving_passengers
        self.time += DOOR_OPEN_TIME + len(leaving_passengers) * ENTER_TIME

    def idle(self):
        self.time += IDLE_TIME
