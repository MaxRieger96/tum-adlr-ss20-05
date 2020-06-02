from enum import Enum

from src.elevator_rl.house import House

from elevator_rl.elevator_env_render import render
from elevator_rl.example_houses import get_10_story_house


class ElevatorActionEnum(Enum):
    DOWN = -1
    OPEN = 0
    UP = 1
    IDLE = 2

    @staticmethod
    def count() -> int:
        return len([d for d in ElevatorActionEnum])


class ElevatorEnvAction:
    def __init__(self, elevator_idx: int, elevator_action: ElevatorActionEnum):
        self.elevator_idx: int = elevator_idx
        self.elevator_action: ElevatorActionEnum = elevator_action


class ElevatorEnv:
    def __init__(self, house: House):
        self.house: House = house
        self.next_elevator: int = 0

    def step(self, env_action: ElevatorEnvAction):
        # move only if valid move
        if env_action.elevator_idx not in range(0, len(self.house.elevators)):
            raise ValueError("Elevator-ID does not exist")
        assert (
            env_action.elevator_idx == self.next_elevator
        ), "elevators should be controlled in the right order"

        # perform action
        if env_action.elevator_action == ElevatorActionEnum.DOWN:
            self.house.elevators[env_action.elevator_idx].down()
        elif env_action.elevator_action == ElevatorActionEnum.UP:
            self.house.elevators[env_action.elevator_idx].up()
        elif env_action.elevator_action == ElevatorActionEnum.OPEN:
            self.house.open_elevator(env_action.elevator_idx)
        elif env_action.elevator_action == ElevatorActionEnum.IDLE:
            self.house.elevators[env_action.elevator_idx].idle()

        # find elevator which performs action next
        self.next_elevator = self.house.next_to_move()

        # run time until action takes place
        self.house.elapse_time(self.house.elevators[self.next_elevator].time)

    def render(self):
        render(self.house)


def main():
    house = get_10_story_house()

    env = ElevatorEnv(house)

    for _ in range(5):
        for _ in range(4):
            action = ElevatorEnvAction(env.next_elevator, ElevatorActionEnum.UP)
            env.step(action)

        env.render()
    print(house)


if __name__ == "__main__":
    main()
