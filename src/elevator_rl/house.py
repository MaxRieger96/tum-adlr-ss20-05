from typing import List
from typing import Set

import numpy as np

from elevator_rl.elevator import Elevator
from elevator_rl.passenger import Passenger
from elevator_rl.passenger_generator import PassengerGenerator


class House:
    def __init__(
        self,
        number_of_floors: int,
        elevator_capacity: int,
        number_of_elevators: int,
        request_rates: np.ndarray,
        target_probabilities: np.ndarray,
    ):
        assert number_of_floors > 1
        assert elevator_capacity > 0
        assert number_of_elevators > 0

        self.number_of_floors: int = number_of_floors
        self.up_requests: np.ndarray = np.zeros(self.number_of_floors, bool)
        self.down_requests: np.ndarray = np.zeros(self.number_of_floors, bool)
        self.up_requests_waiting_since: np.ndarray = np.zeros(
            self.number_of_floors, float
        )
        self.down_requests_waiting_since: np.ndarray = np.zeros(
            self.number_of_floors, float
        )

        self.elevators: List[Elevator] = [
            Elevator(
                elevator_capacity,
                0,
                set(),
                np.zeros(number_of_floors, dtype=bool),
                self,
            )
            for _ in range(number_of_elevators)
        ]

        self.passenger_gen: PassengerGenerator = PassengerGenerator(
            self, request_rates, target_probabilities
        )

        self.time: float = 0

    def elapse_time_to(self, until_time: float):
        # generate signals
        assert until_time >= self.time, "we cannot travel back in time"
        time_delta = until_time - self.time
        self.passenger_gen.create_requests(time_delta)
        self.time = until_time

    def open_elevator(self, elevator_idx: int) -> Set[Passenger]:
        # first people exit the elevator
        assert 0 <= elevator_idx < len(self.elevators)
        elevator = self.elevators[elevator_idx]
        leaving_passengers = {
            p for p in elevator.passengers if p.decide_to_leave_elevator(elevator.floor)
        }
        elevator.passengers_exit_elevator(leaving_passengers)

        # then people enter the elevator
        entering_passengers, new_floor_requests = self.passenger_gen.create_passengers(
            elevator
        )
        elevator.passengers_enter_elevator(entering_passengers, new_floor_requests)
        return leaving_passengers

    def next_to_move(self) -> int:
        return int(np.argmin([e.time for e in self.elevators]))

    def get_expected_passenger_count(self, current_time: float) -> float:
        result = sum([len(e.passengers) for e in self.elevators])
        # adding 1 + the expected number of passengers arrived after the first for each
        # request
        for floor, arrival_time in enumerate(self.up_requests_waiting_since):
            if self.up_requests[floor]:
                result += 1 + (
                    self.passenger_gen.expected_passengers_waiting(
                        floor, arrival_time, current_time
                    )
                )
        for floor, arrival_time in enumerate(self.down_requests_waiting_since):
            if self.down_requests[floor]:
                result += 1 + (
                    self.passenger_gen.expected_passengers_waiting(
                        floor, arrival_time, current_time
                    )
                )

        return result
