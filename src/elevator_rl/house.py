from enum import Enum
from typing import List


class RequestDirection(Enum):
    DOWN = -1
    UP = 1


class PassengerRequest:
    def __init__(self, floor: int, direction: RequestDirection):
        self.floor = floor
        self.direction = direction


class House:
    number_of_floors: int
    passenger_request: List[
        PassengerRequest
    ]  # TODO: maybe store the time of request as well?

    def __init__(
        self, number_of_floors: int, elevator_requests: List[PassengerRequest]
    ):
        self.number_of_floors = number_of_floors
        self.elevator_requests = elevator_requests
