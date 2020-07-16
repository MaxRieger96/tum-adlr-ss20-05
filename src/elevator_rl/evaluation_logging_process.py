import os.path as path
from copy import deepcopy
from multiprocessing import Manager
from multiprocessing.context import Process
from multiprocessing.queues import Queue
from typing import Dict

from torch.utils.tensorboard import SummaryWriter

from elevator_rl.alphazero.model import Model
from elevator_rl.alphazero.sample_generator import Generator
from elevator_rl.alphazero.tensorboard import Logger


class EvaluationLoggingProcess:
    queue: Queue

    def __init__(self, config: Dict, run_name: str):
        manager = Manager()
        self.queue = manager.Queue()
        self.config = config
        self.run_name = run_name
        self.start()

    def start(self):
        p = Process(target=self.logging_loop, args=(self.config, self.run_name))
        p.daemon = True
        p.start()

    def logging_loop(self, config, run_name):
        logger_eval = Logger(
            SummaryWriter(path.join(config["path"], run_name + "_eval"))
        )
        while True:
            log_summary, iteration = self.queue.get()
            batch_count = (
                self.config["train"]["samples_per_iteration"]
                // self.config["train"]["batch_size"]
            )
            logger_eval.log_summary(log_summary, iteration * batch_count, "eval")
            print("Logged Eval")


def evaluation_process(
    generator: Generator,
    config: Dict,
    model: Model,
    i: int,
    run_name: str,
    eval_logging_process: EvaluationLoggingProcess,
):
    # Evaluating Process (depending on visualize_iterations flag outputs a video as well)
    mcts_temp = 0
    observations, pis, total_reward, summary = generator.perform_episode(
        config["mcts"]["samples"],
        mcts_temp,
        config["mcts"]["cpuct"],
        config["mcts"]["observation_weight"],
        deepcopy(model),
        config["visualize_iterations"],
        i,
        run_name,
    )
    eval_logging_process.queue.put((summary, i))
