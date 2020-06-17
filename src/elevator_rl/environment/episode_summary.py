from statistics import stdev
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elevator_rl.environment.elevator_env import ElevatorEnv


class Summary:
    def __init__(
        self,
        nr_passengers_transported: float,
        nr_passengers_waiting: float,
        avg_waiting_time_transported: float,
        elapsed_time: float,
        accumulated_reward: float,
        quadratic_waiting_time: float,
    ):
        self.nr_passengers_transported: float = nr_passengers_transported
        self.nr_passengers_waiting: float = nr_passengers_waiting
        self.avg_waiting_time_transported: float = avg_waiting_time_transported
        self.elapsed_time: float = elapsed_time
        self.accumulated_reward: float = accumulated_reward
        self.quadratic_waiting_time: float = quadratic_waiting_time

    def __str__(self):
        return (
            f"transported {self.nr_passengers_transported} passengers "
            f"({self.percent_transported()*100:.1f}%) "
            f"while {self.nr_passengers_waiting:.1f} are still waiting in a total time "
            f"of {self.elapsed_time:.1f}s\n"
            f"transport took {self.avg_waiting_time_transported:.1f}s on average\n"
            f"quadratic waiting time: \t\t\t\t\t\t{self.quadratic_waiting_time:.1f}\n"
            f"estimated accumulated quadratic waiting time:\t"
            f"{-1*self.accumulated_reward:.1f}\n"
        )

    def percent_transported(self) -> float:
        return self.nr_passengers_transported / (
            self.nr_passengers_transported + self.nr_passengers_waiting
        )


def get_summary(env: "ElevatorEnv") -> Summary:
    """
    gives a summary of the episode performed by this environment
    TODO add more relevant info
    :return: nr_passengers_transported, nr_passengers_waiting
    """
    if len(env.transported_passenger_times) > 0:
        avg_waiting_time_transported = sum(env.transported_passenger_times) / len(
            env.transported_passenger_times
        )
    else:
        avg_waiting_time_transported = 0
    return Summary(
        nr_passengers_transported=len(env.transported_passenger_times),
        nr_passengers_waiting=env.house.get_expected_passenger_count(env.house.time),
        avg_waiting_time_transported=avg_waiting_time_transported,
        elapsed_time=env.house.time,
        accumulated_reward=env.reward_acc,
        quadratic_waiting_time=env.get_quadratic_total_waiting_time(),
    )


def combine_summaries(summaries: List[Summary]) -> Tuple[Summary, Summary]:
    n = len(summaries)
    return (
        Summary(
            nr_passengers_transported=sum(
                s.nr_passengers_transported for s in summaries
            )
            / n,
            nr_passengers_waiting=sum(s.nr_passengers_waiting for s in summaries) / n,
            avg_waiting_time_transported=sum(
                s.avg_waiting_time_transported for s in summaries
            )
            / n,
            elapsed_time=sum(s.elapsed_time for s in summaries) / n,
            accumulated_reward=sum(s.accumulated_reward for s in summaries) / n,
            quadratic_waiting_time=sum(s.quadratic_waiting_time for s in summaries) / n,
        ),
        Summary(
            nr_passengers_transported=stdev(
                s.nr_passengers_transported for s in summaries
            ),
            nr_passengers_waiting=stdev(s.nr_passengers_waiting for s in summaries),
            avg_waiting_time_transported=stdev(
                s.avg_waiting_time_transported for s in summaries
            ),
            elapsed_time=stdev(s.elapsed_time for s in summaries),
            accumulated_reward=stdev(s.accumulated_reward for s in summaries),
            quadratic_waiting_time=stdev(s.quadratic_waiting_time for s in summaries),
        ),
    )
