import numpy as np

from elevator_rl.environment.elevator_env import ElevatorActionEnum


class RandomPolicy:
    def __init__(self):
        pass

    def get_policy_and_value(self):
        value = 1
        return (
            np.array([1 / ElevatorActionEnum.count() for _ in ElevatorActionEnum]),
            value,
        )
