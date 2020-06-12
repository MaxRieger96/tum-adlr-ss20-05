from copy import deepcopy
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from elevator_rl.alphazero.model import Model
from torch.distributions import Categorical

from elevator_rl.alphazero.mcts import MCTS
from elevator_rl.alphazero.ranked_reward import RankedRewardBuffer
from elevator_rl.environment.elevator_env import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.elevator_env import ElevatorEnvAction


class Generator:
    env: ElevatorEnv
    use_ranked_reward: bool
    ranked_reward_buffer: RankedRewardBuffer

    def __init__(
        self, env: ElevatorEnv, ranked_reward_buffer: Optional[RankedRewardBuffer],
    ):
        self.env = env
        self.use_ranked_reward = ranked_reward_buffer is not None
        self.ranked_reward_buffer = ranked_reward_buffer

    @staticmethod
    def sample_action(prior: torch.Tensor) -> int:
        cat = Categorical(prior)
        action = cat.sample()
        return action.numpy().flatten()[0]

    def perform_episode(
        self,
        mcts_samples: int,
        mcts_temp: float,
        mcts_cpuct: int,
        mcts_observation_weight: float,
        model: Model = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        current_env = deepcopy(self.env)
        pis = []
        observations = [current_env.get_observation().as_array()]
        total_reward = 0
        while not current_env.is_end_of_day():
            mcts = MCTS(
                mcts_samples,
                mcts_cpuct,
                self.ranked_reward_buffer,
                mcts_observation_weight,
                model,
            )

            probs = mcts.get_action_probabilities(current_env, mcts_temp)
            probs = np.array(probs, dtype=np.float32)

            pis.append(probs)

            probs = torch.from_numpy(probs)
            action = ElevatorActionEnum(self.sample_action(probs))
            obs, reward = current_env.step(
                ElevatorEnvAction(current_env.next_elevator, action)
            )
            observations.append(obs.as_array())
            total_reward += reward

        print(".", end="", flush=True)
        return observations, pis, total_reward
