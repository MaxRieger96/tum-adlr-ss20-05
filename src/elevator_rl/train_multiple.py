from elevator_rl.train import main


def multiple():
    configs = ["default", "8stores", "8stores_2elevators"]
    for config in configs:
        main(config_name=config)


if __name__ == "__main__":
    multiple()
