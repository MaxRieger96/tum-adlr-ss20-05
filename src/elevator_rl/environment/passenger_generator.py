from typing import List
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np

from elevator_rl.environment.elevator import Elevator
from elevator_rl.environment.passenger import Passenger

if TYPE_CHECKING:
    from elevator_rl.environment.house import House

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def merge_sorted_lists(
    l1a: List[T1], l1b: List[T2], l2a: List[T1], l2b: List[T2]
) -> Tuple[List[T1], List[T2], List[bool]]:
    i = 0
    j = 0
    a = []
    b = []
    c = []
    assert len(l1a) == len(l1b)
    assert len(l2a) == len(l2b)
    while i < len(l1a) or j < len(l2a):
        if i >= len(l1a):
            a.append(l2a[j])
            b.append(l2b[j])
            c.append(False)
            j += 1
        elif j >= len(l2a):
            a.append(l1a[i])
            b.append(l1b[i])
            c.append(True)
            i += 1
        elif l1a[i] < l2a[j]:
            a.append(l1a[i])
            b.append(l1b[i])
            c.append(True)
            i += 1
        else:
            a.append(l2a[j])
            b.append(l2b[j])
            c.append(False)
            j += 1
    return a, b, c


def one_hot(x: int, n: int) -> np.ndarray:
    return np.array([float(i == x) for i in range(n)], dtype=float)


def equal_distribution(indices: Set[int], n: int) -> np.ndarray:
    """
    create an equal distribution for given indices, probability of zero for others
    n specifies length of vector
    :return: probabilities: np.ndarray
    """
    assert len(indices) > 0
    res = np.zeros(n, dtype=float)
    res[list(indices)] = 1.0
    return res / np.sum(res)


def adapted_categorical_distribution_distribution(
    original_distribution: np.ndarray, indices: Set[int]
) -> np.ndarray:
    """
    adapt distribution so that only given indices have probability >=0
    :return: adapted_distribution: np.ndarray
    """
    assert len(indices) > 0
    excluded_indices = {i for i in range(len(original_distribution))} - indices
    res = original_distribution.copy()
    res[list(excluded_indices)] = 0
    return res / np.sum(res)


def index_of_first_true_after_index(bool_list: List[bool], first_index: int) -> int:
    assert first_index < len(bool_list)
    while not bool_list[first_index]:
        first_index += 1
    return first_index


def invert_bool_list(bool_list: List[bool]) -> List[bool]:
    return [not v for v in bool_list]


class PassengerGenerator:
    def __init__(
        self,
        house: "House",
        request_rates: np.ndarray,
        target_probabilities: np.ndarray,
    ):
        self.house: "House" = house
        assert len(request_rates) == house.number_of_floors
        self.request_rates: np.ndarray = request_rates
        assert np.sum(target_probabilities) == 1.0
        assert len(target_probabilities) == house.number_of_floors
        self.target_probabilities: np.ndarray = target_probabilities

    def create_requests(self, time_delta: float):
        # given a time create a number of new passenger requests
        if time_delta == 0:
            return

        # sample number of new requests for each floor
        new_lambdas = self.request_rates * time_delta
        counts = np.random.poisson(new_lambdas)

        for floor, count in enumerate(counts):
            if count == 0:
                continue

            print("new request!")

            # sample arrival time for each passenger
            arrival_times = np.random.uniform(low=0, high=time_delta, size=count)

            # sample target direction for each passenger
            new_target_probabilities = self.target_probabilities.copy()
            new_target_probabilities[floor] = 0
            new_target_probabilities /= np.sum(new_target_probabilities)
            target_floors_one_hot = np.random.multinomial(
                n=1, pvals=new_target_probabilities, size=count
            )
            target_floors = np.array(
                [np.where(r == 1)[0][0] for r in target_floors_one_hot]
            )
            target_dir_is_up = target_floors > floor
            assert target_dir_is_up.shape == arrival_times.shape

            # update request signals
            if np.any(target_dir_is_up) and not self.house.up_requests[floor]:
                self.house.up_requests[floor] = True
                self.house.up_requests_waiting_since[floor] = (
                    np.min(arrival_times[target_dir_is_up]) + self.house.time
                )

            if not np.all(target_dir_is_up) and not self.house.down_requests[floor]:
                self.house.down_requests[floor] = True
                self.house.down_requests_waiting_since[floor] = (
                    np.min(arrival_times[~target_dir_is_up]) + self.house.time
                )

    def _get_target_probs(self, exclude_idxs: List[int]) -> np.ndarray:
        for i in exclude_idxs:
            assert 0 <= i <= len(self.target_probabilities)
        mask = np.ones_like(self.target_probabilities)
        mask[exclude_idxs] = 0
        target_probabilities = self.target_probabilities * mask
        return target_probabilities / np.sum(target_probabilities)

    # TODO: TIM continue reading from here
    def _create_up_or_down_passengers(
        self, floor: int, time: float, exclude_idxs: List[int], waiting_since: float
    ) -> Tuple[np.ndarray, List[int]]:
        time_delta = time - waiting_since
        count = 1
        simulated_time = 0
        rate = self.request_rates[floor]
        arrival_times = [0]

        while True:
            time_to_next = np.random.exponential(1 / rate)
            simulated_time += time_to_next
            if simulated_time <= time_delta:
                count += 1
                arrival_times.append(simulated_time)
            else:
                break

        arrival_times = np.array(arrival_times) + waiting_since
        targets = np.random.multinomial(
            1, self._get_target_probs(exclude_idxs), size=count
        )
        targets = [np.where(r == 1)[0][0] for r in targets]

        return arrival_times, targets

    def _create_up_passengers(
        self, floor: int, time: float
    ) -> Tuple[np.ndarray, List[int]]:
        if self.house.up_requests[floor]:
            not_reachable = [
                i for i in range(len(self.target_probabilities)) if i <= floor
            ]
            waiting_since = self.house.up_requests_waiting_since[floor]
            return self._create_up_or_down_passengers(
                floor, time, not_reachable, waiting_since
            )
        else:
            return np.array([]), []

    def _create_down_passengers(
        self, floor: int, time: float
    ) -> Tuple[np.ndarray, List[int]]:
        if self.house.down_requests[floor]:
            not_reachable = [
                i for i in range(len(self.target_probabilities)) if i >= floor
            ]
            waiting_since = self.house.down_requests_waiting_since[floor]
            return self._create_up_or_down_passengers(
                floor, time, not_reachable, waiting_since
            )
        else:
            return np.array([]), []

    def _update_up_down_requests(
        self, n_get_in: int, arrival_times: List[float], want_up: List[bool], floor: int
    ) -> Tuple[bool, bool]:
        remaining = arrival_times[n_get_in:]

        # remove up / down requests if all passengers got in
        # update up / down waiting since if not all passengers got in
        if len(remaining) == 0:
            all_up_in = True
            all_down_in = True
        else:
            all_up_in = not np.any(want_up[n_get_in:])
            all_down_in = np.all(want_up[n_get_in:])

        if all_up_in:
            self.house.up_requests[floor] = False
        else:
            self.house.up_requests_waiting_since[floor] = arrival_times[
                index_of_first_true_after_index(want_up, n_get_in)
            ]

        if all_down_in:
            self.house.down_requests[floor] = False
        else:
            self.house.down_requests_waiting_since[floor] = arrival_times[
                index_of_first_true_after_index(invert_bool_list(want_up), n_get_in)
            ]

        return all_up_in, all_down_in

    def sample_passenger_times(
        self, floor: int, time: float
    ) -> Tuple[List[float], List[int], List[bool]]:
        if not self.house.up_requests[floor] and not self.house.down_requests[floor]:
            return [], [], []
        else:
            up_arrival_times, up_targets = self._create_up_passengers(floor, time)
            down_arrival_times, down_targets = self._create_down_passengers(floor, time)
            arrival_times, targets, want_up = merge_sorted_lists(
                up_arrival_times, up_targets, down_arrival_times, down_targets
            )
            assert len(arrival_times) == len(targets) == len(want_up)
            return arrival_times, targets, want_up

    def create_passengers(self, elevator: Elevator) -> Tuple[Set[Passenger], List[int]]:
        """
        sample passengers which leave given elevator and passengers which enter
         1. Calc nr of passengers waiting for each dir iteratively (~Exp)
         2. Decide which passengers enter by capacity, omit others
         3. For each passenger draw target floor
         4. Make set of target floors
         5. Assign probabilities of target floors for passengers
              (100% for one passenger on each floor in set of target floors)
        :return: passengers, requests: Tuple[Set[Passenger], List[int]]
        """
        if (
            not self.house.up_requests[elevator.floor]
            and not self.house.down_requests[elevator.floor]
        ):
            return set(), []
        else:
            # get arrival times for each passenger
            arrival_times, targets, want_up = self.sample_passenger_times(
                elevator.floor, elevator.time
            )
            n_new_passengers = min(elevator.free_places(), len(arrival_times))

            # check if all passengers got in and update request signals
            self._update_up_down_requests(
                n_new_passengers, arrival_times, want_up, floor=elevator.floor
            )

            # get target floors for each passenger
            up_passengers = [i for i in range(n_new_passengers) if want_up[i]]
            up_floors = {targets[i] for i in up_passengers}
            down_passengers = [i for i in range(n_new_passengers) if not want_up[i]]
            down_floors = {targets[i] for i in down_passengers}

            requests = list(up_floors | down_floors)

            passengers = []
            # assign 100% to a passenger for each floor
            for i, floor in enumerate(up_floors):
                p = Passenger(
                    one_hot(floor, len(self.target_probabilities)),
                    arrival_times[up_passengers[i]],
                )
                passengers.append(p)
            for i, floor in enumerate(down_floors):
                p = Passenger(
                    one_hot(floor, len(self.target_probabilities)),
                    arrival_times[down_passengers[i]],
                )
                passengers.append(p)

            # assign random distributions to remaining passengers
            for i in range(len(up_floors), len(up_passengers)):
                up_distribution = adapted_categorical_distribution_distribution(
                    self.target_probabilities, up_floors
                )
                p = Passenger(up_distribution, arrival_times[up_passengers[i]])
                passengers.append(p)

            for i in range(len(down_floors), len(down_passengers)):
                down_distribution = adapted_categorical_distribution_distribution(
                    self.target_probabilities, down_floors
                )
                p = Passenger(down_distribution, arrival_times[down_passengers[i]])
                passengers.append(p)

            return set(passengers), requests

    def expected_passengers_waiting(
        self, floor: int, waiting_since: float, current_time: float
    ) -> float:
        """
        calculate the expected number of passengers appearing at the given floor between
        waiting_since and current_time
        :return: expected_nr_waiting: float
        """
        # assuming the poisson distribution, the expected value is exactly the rate
        time_delta = current_time - waiting_since
        return self.request_rates[floor] * time_delta
