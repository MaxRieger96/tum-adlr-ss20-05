import math
from copy import deepcopy
from typing import List
from typing import Optional

import numpy as np
import torch
from elevator_rl.alphazero.model import Model

from elevator_rl.alphazero.ranked_reward import RankedRewardBuffer
from elevator_rl.baseline.uniform_model import UniformModel
from elevator_rl.environment.elevator_env import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.elevator_env import ElevatorEnvAction

EPS = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_reward(x: float, k: float) -> float:
    return np.tanh(x / k)


class MCTS:
    def __init__(
        self,
        num_simulations: int,
        c_puct: float,
        ranked_reward_buffer: Optional[RankedRewardBuffer],
        observation_weight: float,
        model: Model,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.ranked_reward_buffer = ranked_reward_buffer
        assert 0 <= observation_weight <= 1, "observation weight must be in [0, 1]"
        self.observation_weight = observation_weight

        self.action_value = {}  # stores action values Q
        self.visit_count = {}  # stores #times edge s,a was visited
        self.visit_count_state = {}  # stores #times state was visited
        self.prior_prob_state = {}  # stores prior of policy
        self.all_states_dump = []

        self.neural_network_mcts = True if model is not None else False
        if model is None:
            self.model = UniformModel()
        else:
            self.model = model
            self.model.eval()

    def get_action_probabilities(
        self, current_env: ElevatorEnv, temperature: float
    ) -> List[float]:

        for i in range(self.num_simulations):
            new_env = deepcopy(current_env)
            self.search(new_env, prev_reward=current_env.reward_acc, step_depth=1)

        state = current_env.get_observation().to_hashable()
        count_taken_directions = [
            self.visit_count[(state, action)]
            if (state, action) in self.visit_count
            else 0
            for action in ElevatorActionEnum
        ]

        if temperature == 0:
            best_direction = np.argmax(count_taken_directions)
            probs = [0] * len(count_taken_directions)
            probs[int(best_direction)] = 1
            return probs

        count_taken_directions_with_temperature = [
            x ** (1.0 / temperature) for x in count_taken_directions
        ]
        counts_sum = float(sum(count_taken_directions_with_temperature))
        probs = [x / counts_sum for x in count_taken_directions_with_temperature]
        return probs

    def search(
        self, current_env: ElevatorEnv, prev_reward: float, step_depth: int,
    ) -> float:

        state = current_env.get_observation().to_hashable()
        self.all_states_dump.append(current_env.get_observation().as_array())
        if current_env.is_end_of_day():
            # terminal node
            if self.ranked_reward_buffer is not None:
                return self.ranked_reward_buffer.get_ranked_reward(
                    current_env.reward_acc
                )
            else:
                return current_env.reward_acc

        valid_actions = current_env.house.elevators[
            current_env.next_elevator
        ].valid_actions()
        if state not in self.prior_prob_state:
            # leaf node
            if self.neural_network_mcts:
                observation_array = current_env.get_observation().as_array()
                x = torch.from_numpy(observation_array)
                x = x.to(device).to(torch.float32)
                policy, value = self.model(x)
                policy = policy.squeeze().cpu().detach().numpy()
                value = value.squeeze().cpu().detach().numpy()
            else:
                policy, value = self.model.get_policy_and_value(current_env)
            self.prior_prob_state[state] = policy

            self.prior_prob_state[state] = (
                self.prior_prob_state[state] * valid_actions
            )  # masking invalid moves
            sum_prior_probs = np.sum(self.prior_prob_state[state])
            if sum_prior_probs > 0:
                self.prior_prob_state[state] /= sum_prior_probs  # renormalize
            else:
                raise ValueError(
                    "There is no valid move left, this should never happen"
                )

            self.visit_count_state[state] = 0
            avg_observed_reward = (current_env.reward_acc - prev_reward) / step_depth
            combined = (
                self.observation_weight * normalize_reward(avg_observed_reward, 10)
                + (1 - self.observation_weight) * value
            )
            return combined

        cur_best = -float("inf")
        best_action = None

        # pick the action with the highest upper confidence bound
        for action in ElevatorActionEnum:
            if valid_actions[ElevatorActionEnum(action).value]:
                if (state, action) in self.action_value:
                    upper_bound = self.action_value[
                        (state, action)
                    ] + self.c_puct * self.prior_prob_state[state][
                        action.value
                    ] * math.sqrt(
                        self.visit_count_state[state]
                    ) / (
                        1 + self.visit_count[(state, action)]
                    )
                else:
                    upper_bound = (
                        self.c_puct
                        * self.prior_prob_state[state][action.value]
                        * math.sqrt(self.visit_count_state[state] + EPS)
                    )  # Q = 0 ?

                if upper_bound > cur_best:
                    cur_best = upper_bound
                    best_action = action

        action = best_action
        # TODO: decide: do we need to store/regard action or elevator_action as the "action"
        elevator_action = ElevatorEnvAction(
            current_env.next_elevator, ElevatorActionEnum(best_action)
        )
        observation, reward = current_env.step(elevator_action)

        value = self.search(current_env, prev_reward, step_depth=(step_depth + 1))

        if (state, action) in self.action_value:
            self.action_value[(state, action)] = (
                self.visit_count[(state, action)] * self.action_value[(state, action)]
                + value
            ) / (self.visit_count[(state, action)] + 1)
            self.visit_count[(state, action)] += 1

        else:
            self.action_value[(state, action)] = value
            self.visit_count[(state, action)] = 1

        self.visit_count_state[state] += 1
        return value
