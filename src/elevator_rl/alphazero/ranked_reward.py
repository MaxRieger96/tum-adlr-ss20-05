from random import getrandbits
from typing import List
from typing import Optional


class RankedRewardBuffer:
    def __init__(self, capacity: int, threshold: float):
        self.capacity: int = capacity
        self.memory: List[Optional[float]] = []
        self.position: int = 0
        self.threshold: float = threshold

    def __len__(self):
        return len(self.memory)

    def push(self, reward: float) -> float:
        """Saves a train sample"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = reward
        self.position = (self.position + 1) % self.capacity
        return self.get_ranked_reward(reward)

    def get_ranked_reward(self, reward: float):
        if len(self.memory) == 0:
            if bool(getrandbits(1)):
                return 1
            else:
                return -1

        nr_of_worse_results = sum(1 for v in self.memory if v < reward)
        nr_of_equal_results = sum(1 for v in self.memory if v == reward)

        fraction_of_worse_results = nr_of_worse_results / len(self.memory)
        fraction_of_equal_results = nr_of_equal_results / len(self.memory)

        if fraction_of_worse_results >= self.threshold:
            return 1
        elif fraction_of_worse_results + fraction_of_equal_results >= self.threshold:
            return 0
        else:
            return -1
