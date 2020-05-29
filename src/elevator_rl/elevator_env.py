from src.elevator_rl.elevator import Elevator
from src.elevator_rl.elevator import ElevatorActionEnum
from src.elevator_rl.elevator_env_render import render
from src.elevator_rl.house import House


class ElevatorEnvAction:
    def __init__(self, elevator_idx: int, elevator_action: ElevatorActionEnum):
        self.elevator_idx = elevator_idx
        self.elevator_action = elevator_action


class ElevatorEnv:
    def __init__(self, house: House, nr_elevators: int, elevator_capacity: int):
        self.house = house
        self.elevators = [
            Elevator(elevator_capacity, 0, 0, []) for _ in range(nr_elevators)
        ]

    def check_passenger_requests(self):
        pass

    def step(self, env_action: ElevatorEnvAction):
        self.check_passenger_requests()

        if env_action.elevator_idx not in range(0, len(self.elevators)):
            raise ValueError("Elevator-ID does not exist")
        # Move only if valid move
        if not (
            self.elevators[env_action.elevator_idx].floor == 0
            and env_action.elevator_action == ElevatorActionEnum.DOWN
        ) and not (
            self.elevators[env_action.elevator_idx].floor == self.house.number_of_floors
            and env_action.elevator_action == ElevatorActionEnum.UP
        ):
            self.elevators[
                env_action.elevator_idx
            ].floor += env_action.elevator_action.value
        if env_action.elevator_action == ElevatorActionEnum.STAY:
            self.elevators[env_action.elevator_idx].passenger_exchange(
                self.house.elevator_requests
            )
        pass

    def render(self):
        render(self.house, self.elevators)


def main():
    env = ElevatorEnv(House(8, []), 3, 10)
    active_elevator = 0  # for now only one elevator moving
    env.render()
    while 0 == 0:
        direction = int(input("Direction"))
        if direction == -1:
            env.step(ElevatorEnvAction(active_elevator, ElevatorActionEnum.DOWN))
        elif direction == 0:
            env.step(ElevatorEnvAction(active_elevator, ElevatorActionEnum.STAY))
        else:
            env.step(ElevatorEnvAction(active_elevator, ElevatorActionEnum.UP))
        env.render()


if __name__ == "__main__":
    main()
