from copy import deepcopy
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import os
from pathlib import Path
import torch
import subprocess
from torch.distributions import Categorical
from torch.multiprocessing import Pool
from torch.multiprocessing import set_start_method

from elevator_rl.alphazero.mcts import MCTS
from elevator_rl.alphazero.model import Model
from elevator_rl.alphazero.ranked_reward import RankedRewardBuffer
from elevator_rl.environment.elevator_env import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.elevator_env import ElevatorEnvAction
from elevator_rl.environment.episode_summary import Summary
from elevator_rl.environment.observation import ObservationType

try:
    set_start_method("spawn")
except RuntimeError:
    pass


class Generator:
    env: ElevatorEnv
    ranked_reward_buffer: RankedRewardBuffer

    def __init__(
            self, env: ElevatorEnv, ranked_reward_buffer: Optional[RankedRewardBuffer],
    ):
        self.env = env
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
            model: Model,
            render: bool = False,
            iteration: int = None,
            run_name: str = None
    ) -> Tuple[List[ObservationType], List[np.ndarray], int, Summary]:
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
            prev_time = current_env.house.time
            action = ElevatorActionEnum(self.sample_action(probs))
            env_action = ElevatorEnvAction(current_env.next_elevator, action)
            obs, reward = current_env.step(env_action)
            observations.append(obs.as_array())
            total_reward += reward

            if render:
                root_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(root_dir,
                                    "{}/../plots/run_{}/iteration{}".format(root_dir,
                                                                            run_name,
                                                                            iteration))
                Path(path).mkdir(parents=True, exist_ok=True)
                current_env.render(method="file", prev_time=prev_time, path=path,
                                   action=env_action)

        print(".", end="", flush=True)
        if render:
            subprocess.Popen(["./animate.sh", "-r", run_name, "-i", str(iteration)])
        return observations, pis, total_reward, current_env.get_summary()


class EpisodeFactory:
    def __init__(self, generator: Generator):
        self._generator: Generator = generator

    def create_episodes(
            self,
            n_episodes: int,
            n_processes: int,
            mcts_samples: int,
            mcts_temp: float,
            mcts_cpuct: int,
            mcts_observation_weight: float,
            model: Model,
    ):
        pool = Pool(n_processes)
        res = pool.starmap(
            self._generator.perform_episode,
            [[mcts_samples, mcts_temp, mcts_cpuct, mcts_observation_weight, model]]
            * n_episodes,
        )
        pool.close()
        pool.terminate()
        pool.join()
        return res


# TODO create async episode factory
