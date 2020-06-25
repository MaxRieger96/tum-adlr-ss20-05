import random
from typing import List
from typing import Tuple

import numpy as np

from elevator_rl.environment.observation import ObservationType

# sample: (observation, pi, z)

Sample = Tuple[ObservationType, np.ndarray, float]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: List[Sample] = []
        self.position: int = 0

    def push(self, sample: Sample):
        """Saves a train sample"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Sample]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
