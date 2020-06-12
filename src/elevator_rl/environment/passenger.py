import logging
from random import random

import numpy as np

EPS = 10e-5


class Passenger:
    def __init__(self, target_floor_distribution: np.ndarray, waiting_since: float):
        assert 1.0 - EPS <= np.sum(target_floor_distribution) <= 1.0 + EPS
        assert np.min(target_floor_distribution) >= 0
        self.target_floor_distribution: np.ndarray = target_floor_distribution
        self.waiting_since: float = waiting_since

    def decide_to_leave_elevator(self, floor: int) -> bool:
        if random() <= self.target_floor_distribution[floor]:
            logging.info("one leaves")
            return True
        else:
            self.target_floor_distribution[floor] = 0
            self.target_floor_distribution /= np.sum(self.target_floor_distribution)
