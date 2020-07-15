from statistics import stdev
from typing import Callable
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
        avg_waiting_time_per_person: float,
        elapsed_time: float,
        accumulated_reward: float,
        quadratic_waiting_time: float,
        waiting_time: float,
        percent_transported: float,
    ):
        self.nr_passengers_transported: float = nr_passengers_transported
        self.nr_passengers_waiting: float = nr_passengers_waiting
        self.avg_waiting_time_transported: float = avg_waiting_time_transported
        self.avg_waiting_time_per_person: float = avg_waiting_time_per_person
        self.elapsed_time: float = elapsed_time
        self.accumulated_reward: float = accumulated_reward
        self.quadratic_waiting_time: float = quadratic_waiting_time
        self.waiting_time: float = waiting_time
        self.percent_transported: float = percent_transported

    def __str__(self):
        return (
            f"transported {self.nr_passengers_transported:.1f} passengers "
            f"({self.percent_transported * 100:.1f}%) "
            f"while {self.nr_passengers_waiting:.1f} are still waiting in a total time "
            f"of {self.elapsed_time:.1f}s\n"
            f"transport took {self.avg_waiting_time_transported:.1f}s on average\n"
            f"quadratic waiting time: \t\t\t\t\t\t{self.quadratic_waiting_time:.1f}\n"
            f"estimated accumulated quadratic waiting time:\t"
            f"{-1 * self.accumulated_reward:.1f}\n"
            f"avg waiting time per person: {self.avg_waiting_time_per_person:.1f}"
        )

    def __abs__(self) -> "Summary":
        return Summary(
            nr_passengers_transported=abs(self.nr_passengers_transported),
            nr_passengers_waiting=abs(self.nr_passengers_waiting),
            avg_waiting_time_transported=abs(self.avg_waiting_time_transported),
            avg_waiting_time_per_person=abs(self.avg_waiting_time_per_person),
            elapsed_time=abs(self.elapsed_time),
            accumulated_reward=abs(self.accumulated_reward),
            quadratic_waiting_time=abs(self.quadratic_waiting_time),
            waiting_time=abs(self.waiting_time),
            percent_transported=abs(self.percent_transported),
        )

    def __add__(self, other: "Summary") -> "Summary":
        return Summary(
            nr_passengers_transported=self.nr_passengers_transported
            + other.nr_passengers_transported,
            nr_passengers_waiting=self.nr_passengers_waiting
            + other.nr_passengers_waiting,
            avg_waiting_time_transported=self.avg_waiting_time_transported
            + other.avg_waiting_time_transported,
            avg_waiting_time_per_person=self.avg_waiting_time_per_person
            + other.avg_waiting_time_per_person,
            elapsed_time=self.elapsed_time + other.elapsed_time,
            accumulated_reward=self.accumulated_reward + other.accumulated_reward,
            quadratic_waiting_time=self.quadratic_waiting_time
            + other.quadratic_waiting_time,
            waiting_time=self.waiting_time + other.waiting_time,
            percent_transported=self.percent_transported + other.percent_transported,
        )

    def __sub__(self, other: "Summary") -> "Summary":
        return Summary(
            nr_passengers_transported=self.nr_passengers_transported
            - other.nr_passengers_transported,
            nr_passengers_waiting=self.nr_passengers_waiting
            - other.nr_passengers_waiting,
            avg_waiting_time_transported=self.avg_waiting_time_transported
            - other.avg_waiting_time_transported,
            avg_waiting_time_per_person=self.avg_waiting_time_per_person
            - other.avg_waiting_time_per_person,
            elapsed_time=self.elapsed_time - other.elapsed_time,
            accumulated_reward=self.accumulated_reward - other.accumulated_reward,
            quadratic_waiting_time=self.quadratic_waiting_time
            - other.quadratic_waiting_time,
            waiting_time=self.waiting_time - other.waiting_time,
            percent_transported=self.percent_transported - other.percent_transported,
        )


class SummaryStdDev(Summary):
    def __str__(self):
        return (
            f"transported passengers stddev: {self.nr_passengers_transported:.1f}\n"
            f"still waiting stddev: {self.nr_passengers_waiting:.1f}\n"
            f"transport time stddev {self.avg_waiting_time_transported:.1f}\n"
            f"quadratic waiting time stddev: {self.quadratic_waiting_time:.1f}\n"
            f"estimated accumulated quadratic waiting time stddev: "
            f"{-1 * self.accumulated_reward:.1f}\n"
        )


def get_summary(env: "ElevatorEnv") -> Summary:
    """
    gives a summary of the episode performed by this environment
    TODO add more relevant info
    :return: nr_passengers_transported, nr_passengers_waiting
    """
    nr_transported = len(env.transported_passenger_times)

    if nr_transported > 0:
        avg_waiting_time_transported = (
            sum(env.transported_passenger_times) / nr_transported
        )
    else:
        avg_waiting_time_transported = 0

    total_passengers = len(
        env.transported_passenger_times
    ) + env.house.get_expected_passenger_count(env.house.time)

    still_waiting_waiting_times = (
        env.house.get_waiting_time_for_all_waiting_passengers()
    )

    avg_waiting_time_per_person = (
        avg_waiting_time_transported * nr_transported + sum(still_waiting_waiting_times)
    ) / total_passengers

    nr_waiting = env.house.get_expected_passenger_count(env.house.time)

    return Summary(
        nr_passengers_transported=nr_transported,
        nr_passengers_waiting=nr_waiting,
        avg_waiting_time_transported=avg_waiting_time_transported,
        avg_waiting_time_per_person=avg_waiting_time_per_person,
        elapsed_time=env.house.time,
        accumulated_reward=env.reward_acc,
        quadratic_waiting_time=env.get_quadratic_total_waiting_time(),
        waiting_time=env.get_total_waiting_time(),
        percent_transported=nr_transported / (nr_transported + nr_waiting),
    )


def combine_summaries(summaries: List[Summary]) -> Tuple[Summary, Summary]:
    return (
        accumulate_summaries(summaries, lambda x: sum(x) / len(x)),
        accumulate_summaries(summaries, lambda x: stdev(x)),
    )


def accumulate_summaries(
    summaries: List[Summary], accumulator: Callable[[List[float]], float]
) -> Summary:
    return Summary(
        nr_passengers_transported=accumulator(
            [s.nr_passengers_transported for s in summaries]
        ),
        nr_passengers_waiting=accumulator([s.nr_passengers_waiting for s in summaries]),
        avg_waiting_time_transported=accumulator(
            [s.avg_waiting_time_transported for s in summaries]
        ),
        avg_waiting_time_per_person=accumulator(
            [s.avg_waiting_time_per_person for s in summaries]
        ),
        elapsed_time=accumulator([s.elapsed_time for s in summaries]),
        accumulated_reward=accumulator([s.accumulated_reward for s in summaries]),
        quadratic_waiting_time=accumulator(
            [s.quadratic_waiting_time for s in summaries]
        ),
        waiting_time=accumulator([s.waiting_time for s in summaries]),
        percent_transported=accumulator([s.percent_transported for s in summaries]),
    )
