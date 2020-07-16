from datetime import datetime
from datetime import timedelta
from random import choice
from random import randint
from random import uniform
from typing import Dict

from elevator_rl.train import learning_loop
from elevator_rl.yparams import YParams


def fill_with_random_values(config: Dict):
    config["mcts"]["samples"] = choice([10, 60, 110]) + randint(0, 50)  # 10 - 160
    config["train"]["episodes"] = choice([8, 16, 32, 64])
    config["replay_buffer"]["size"] = choice([1000, 10000]) * randint(1, 10)
    config["train"]["weight_decay"] = choice([1e-5, 1e-4, 1e-3]) * uniform(0, 10)
    config["train"]["lr"] = choice([1e-5, 1e-4, 1e-3]) * uniform(0, 10)
    config["train"]["batch_size"] = 2 ** randint(3, 10)  # 8 - 1024
    config["train"]["samples_per_iteration"] = 2 ** randint(7, 13)  # 128 - 8192
    config["ranked_reward"]["update_rank"] = choice([True, False])


def main():
    config_name = "default"

    yparams = YParams("config.yaml", config_name)
    config = yparams.hparams
    assert not (
        config["offline_training"] and not config["pretrained_path"]
    ), "Offline training requires pretrained buffer"

    while True:
        fill_with_random_values(config)
        run_name = (
            f'{datetime.now().strftime("%Y-%m-%d_%H:%M")}_S_'
            f'{config["mcts"]["samples"]}MCTS_'
            f'{config["train"]["episodes"]}Ep_'
            f'{config["train"]["samples_per_iteration"]}Samp_'
            f'{config["train"]["lr"]:.1e}LR_'
            f'{config["train"]["weight_decay"]:.1e}Dec'
        )
        print(run_name)
        learning_loop(config, run_name, yparams, timedelta(hours=3))


if __name__ == "__main__":
    main()
