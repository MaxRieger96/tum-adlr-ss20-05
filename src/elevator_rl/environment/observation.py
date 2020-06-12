import numpy as np

from elevator_rl.environment.house import House


class Observation:
    def __init__(self, house: House, next_elevator_idx: int):
        self.number_of_floors: int = house.number_of_floors
        self.elevator_capacity: int = house.elevator_capacity
        self.down_requests: np.ndarray = house.down_requests
        self.up_requests: np.ndarray = house.up_requests
        self.down_requests_waiting_since: np.ndarray = house.down_requests_waiting_since
        self.up_requests_waiting_since: np.ndarray = house.up_requests_waiting_since

        elevators_obs: np.ndarray = np.zeros(0)
        next_elevator: np.ndarray = np.zeros(0)
        for elevator_idx, elevator in enumerate(house.elevators):
            passengers_waiting_time = [
                passenger.waiting_since for passenger in elevator.passengers
            ]
            elevator_obs = np.concatenate(
                [
                    [elevator.floor, elevator.time, len(passengers_waiting_time)],
                    elevator.floor_requests.astype(dtype=int),
                ]
            )
            # TODO: add individual passengers waiting times to observation
            #  (problem: size changes)
            if next_elevator_idx == elevator_idx:
                next_elevator = elevator_obs
            elevators_obs = np.append(elevators_obs, elevator_obs)
        self.next_elevator = next_elevator
        self.elevators_obs = elevators_obs

    def as_array(self) -> np.ndarray:
        temp = np.concatenate(
            [
                np.array([self.number_of_floors]),
                np.array([self.elevator_capacity]),
                self.down_requests.astype(dtype=int),
                self.up_requests.astype(dtype=int),
                self.down_requests_waiting_since,
                self.up_requests_waiting_since,
                self.elevators_obs,
                self.next_elevator,
            ]
        )
        return temp

    def to_hashable(self):
        return tuple(self.as_array())
