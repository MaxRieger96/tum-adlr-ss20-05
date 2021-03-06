from elevator_rl.environment.elevator_env import ElevatorActionEnum
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.elevator_env import ElevatorEnvAction
from elevator_rl.environment.example_houses import get_simple_house


def main():
    house = get_simple_house()

    env = ElevatorEnv(house)
    env.render()

    while True:
        x = input()
        if x == "u":
            print("up")
            action = ElevatorActionEnum.UP
        elif x == "d":
            print("down")
            action = ElevatorActionEnum.DOWN
        elif x == "o":
            print("open")
            action = ElevatorActionEnum.OPEN
        elif x == "i":
            print("idle")
            action = ElevatorActionEnum.IDLE
        else:
            print("valid commands are: {u, d, o,i }")
            continue

        obs, reward = env.step(ElevatorEnvAction(env.next_elevator, action))
        # as_vector = obs.as_array()
        print(
            f"reward: {reward}\t" f"total waiting time: {env.get_total_waiting_time()}"
        )
        env.render()


if __name__ == "__main__":
    main()
