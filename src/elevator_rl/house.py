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
        """
        lets time of the house pass
        this generates request signals of passengers appearing at floors
        """
        # generate signals
        assert until_time >= self.time, "we cannot travel back in time"
        time_delta = until_time - self.time
        self.passenger_gen.create_requests(time_delta)
        self.time = until_time

    def open_elevator(self, elevator_idx: int) -> Set[Passenger]:
        """
        lets people leave the specified elevator, then materializes the passengers at
        the floor based on the requests and lets enter as many passengers as fit in the
        elevator
        :return: leaving_passengers: Set[Passenger]
        """
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
        """
        gives the index of the next elevator, which needs to move
        this is the elevator with the smallest time value

        :return: index: int
        """
        # return int(np.argmin([e.time for e in self.elevators]))
        return min(range(len(self.elevators)), key=lambda x: self.elevators[x].time)

    def get_expected_passenger_count(self, current_time: float) -> float:
        """
        calculates the expected number of passengers waiting at the given time
        the arrival times for passengers at floors are randomly sampled

        :return: count: float
        """
        result = sum([len(e.passengers) for e in self.elevators])
        # adding 1 + the expected number of passengers arrived after the first for each
        # request
        for floor, request in enumerate(self.up_requests):
            if request:
                result += 1 + (
                    self.passenger_gen.expected_passengers_waiting(
                        floor, self.up_requests_waiting_since[floor], current_time
                    )
                )
        for floor, request in enumerate(self.down_requests):
            if request:
                result += 1 + (
                    self.passenger_gen.expected_passengers_waiting(
                        floor, self.down_requests_waiting_since[floor], current_time
                    )
                )

        return result

    def get_arrival_time_for_all_waiting_passengers(self) -> List[float]:
        """
        creates a list of arrival times for all passengers which are currently waiting
        at floors or in elevators
        the arrival times for passengers at floors are randomly sampled

        :return individual_arrival_times: List[float]
        """
        individual_arrival_times = []
        # passengers in elevators
        for e in self.elevators:
            for p in e.passengers:
                individual_arrival_times.append(p.waiting_since)
        # passengers waiting at floors
        for floor, _ in enumerate(self.up_requests):
            times, _, _ = self.passenger_gen.sample_passenger_times(floor, self.time)
            individual_arrival_times += times
        return individual_arrival_times

    def get_waiting_time_for_all_waiting_passengers(self) -> List[float]:
        """for all passengers which are currently waiting
        at floors or in elevators
        the arrival times for passengers at floors are randomly sampled

        :return individual_waiting_times: List[float]
        """
        return [
            self.time - t for t in self.get_arrival_time_for_all_waiting_passengers()
        ]
