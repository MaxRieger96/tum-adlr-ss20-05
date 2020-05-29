from enum import Enum
from typing import List

from src.elevator_rl.house import PassengerRequest


class ElevatorActionEnum(Enum):
    DOWN = -1
    STAY = 0  # TODO: Check if we instead want to have actions such as open/close door
    UP = 1

    @staticmethod
    def count() -> int:
        return len([d for d in ElevatorActionEnum])


class Elevator:
    def __init__(
        self,
        elevator_capacity: int,
        floor: int,
        passengers: int,
        floor_requests: List[int],
    ):
        self.elevator_capacity = elevator_capacity
        self.floor = floor
        self.passengers = (
            passengers  # TODO: not sure if it is reasonable to assume that we know this
        )
        self.floor_requests = floor_requests

    def passenger_exchange(self, elevator_requests: List[PassengerRequest]):
        # TODO: sample passengers that are at this floor and want to get in
        # TODO sample passengers that are in the elevator and want to get out
        if self.floor in self.floor_requests:
            self.passengers -= (
                1  # For now I assume that we only have one person every time
            )
            # TODO issue that floor_requests are not bound to passengers
            #  --> only know that someone wants to leave
            self.floor_requests.remove(self.floor)
        for request in elevator_requests:
            if request.floor == self.floor:
                self.passengers += (
                    1  # For now I assume that we only have one person every time
                )
                elevator_requests.remove(request)
