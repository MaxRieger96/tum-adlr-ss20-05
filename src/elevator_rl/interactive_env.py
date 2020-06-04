from elevator_rl.elevator_env import ElevatorActionEnum
from elevator_rl.elevator_env import ElevatorEnv
from elevator_rl.elevator_env import ElevatorEnvAction
from elevator_rl.example_houses import get_simple_house


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

        print(env.step(ElevatorEnvAction(env.next_elevator, action)))
        env.render()


if __name__ == "__main__":
    main()
