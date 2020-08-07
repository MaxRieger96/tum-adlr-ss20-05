from elevator_rl.train import main


def multiple():
    # configs = ["default", "8stores", "8stores_2elevators"]
    configs = [
        "2elev_3floor",
        "1elev_5floor",
        "2elev_5floor",
        "1elev_7floor",
        "2elev_7floor",
    ]
    for config in configs:
        main(config_name=config)


if __name__ == "__main__":
    multiple()
