from statistics import stdev
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from elevator_rl.environment.episode_summary import accumulate_summaries
from elevator_rl.environment.episode_summary import Summary
from elevator_rl.yparams import YParams


class Logger:
    def __init__(self, writer: SummaryWriter):
        self.writer: SummaryWriter = writer
        self.summaries: List[Tuple[Summary, Summary, Summary]] = []

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
        stdev_summary = abs(accumulate_summaries(summaries, lambda x: stdev(x)))
        upper_stdev_summary = avg_summary + stdev_summary
        lower_stdev_summary = avg_summary - stdev_summary

        self.log_summary(avg_summary, index, "avg")
        self.log_summary(upper_stdev_summary, index, "upper")
        self.log_summary(lower_stdev_summary, index, "lower")

        self.summaries.append((avg_summary, upper_stdev_summary, lower_stdev_summary))

    def plot_summaries(self, show: bool, index: int):
        x = []
        avg_s = []
        upper_s = []
        lower_s = []

        for i, s in enumerate(self.summaries):
            avg_summary, upper_stdev_summary, lower_stdev_summary = s
            x.append(i)
            avg_s.append(avg_summary)
            upper_s.append(upper_stdev_summary)
            lower_s.append(lower_stdev_summary)

        # plot avg waiting time per person
        for metric in (
            "avg_waiting_time_per_person",
            "quadratic_waiting_time",
            "percent_transported",
            "waiting_time",
        ):
            ax = plt.gca()
            y = [vars(s)[metric] for s in avg_s]
            y_upper = [vars(s)[metric] for s in upper_s]
            y_lower = [vars(s)[metric] for s in lower_s]
            ax.plot(x, y)
            ax.fill_between(x, y_lower, y_upper, color="b", alpha=0.1)

            ax.set_title(metric)
            ax.set_xlabel("iterations")
            ax.set_ylabel(metric)
            plt.gcf().set_size_inches(9, 6)

            if show:
                plt.show()
            else:
                fig = plt.gcf()
                # fig.tight_layout(pad=0)
                fig.canvas.draw()
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                self.writer.add_image(metric, data, index, dataformats="HWC")
