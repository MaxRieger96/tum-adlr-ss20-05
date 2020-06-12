import numpy as np

from elevator_rl.environment.elevator_env import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv


class UniformModel:
    def get_policy_and_value(self, env: ElevatorEnv):
        value = 1
        policy = np.array([1 / ElevatorActionEnum.count() for _ in ElevatorActionEnum])
        valid_actions = env.house.elevators[env.next_elevator].valid_actions()
        policy = policy * valid_actions  # masking invalid moves
        policy /= np.sum(policy)  # renormalize
        return policy, value
