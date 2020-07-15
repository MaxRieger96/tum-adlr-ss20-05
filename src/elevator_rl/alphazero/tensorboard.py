from statistics import stdev
from typing import List

from torch.utils.tensorboard import FileWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from elevator_rl.environment.episode_summary import accumulate_summaries
from elevator_rl.environment.episode_summary import Summary
from elevator_rl.yparams import YParams


class Logger:
    def __init__(self, writer: SummaryWriter):
        self.writer: SummaryWriter = writer
        self.filewriter: FileWriter = writer.file_writer

    def write_hparams(self, yparams: YParams):
        exp, ssi, sei = hparams(yparams.flatten(yparams.hparams), {})
        self.writer.file_writer.add_summary(exp)
        self.writer.file_writer.add_summary(ssi)
        self.writer.file_writer.add_summary(sei)

    def log_train(self, logs):
        for log in logs:
            self.writer.add_scalar(*log)

    def log_summary(self, summary: Summary, index: int, name: str):
        name = name + "_"
        self.writer.add_scalar(
            name + "quadratic_waiting_time", summary.quadratic_waiting_time, index
        )
        self.writer.add_scalar(name + "waiting_time", summary.waiting_time, index)
        self.writer.add_scalar(
            name + "percent_transported", summary.percent_transported, index
        )
        self.writer.add_scalar(
            name + "avg_waiting_time_transported",
            summary.avg_waiting_time_transported,
            index,
        )
        self.writer.add_scalar(
            name + "avg_waiting_time_per_person",
            summary.avg_waiting_time_per_person,
            index,
        )
        self.writer.add_scalar(
            name + "nr_waiting", summary.nr_passengers_waiting, index
        )
        self.writer.add_scalar(
            name + "nr_transported", summary.nr_passengers_transported, index
        )

    def write_episode_summaries(self, summaries: List[Summary], index: int):
        avg_summary = accumulate_summaries(summaries, lambda x: sum(x) / len(x))
        stdev_summary = accumulate_summaries(summaries, lambda x: stdev(x))
        upper_stdev_summary = avg_summary + abs(stdev_summary)
        lower_stdev_summary = avg_summary - abs(stdev_summary)

        self.log_summary(avg_summary, index, "avg")
        self.log_summary(upper_stdev_summary, index, "upper")
        self.log_summary(lower_stdev_summary, index, "lower")
