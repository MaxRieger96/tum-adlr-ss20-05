from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from src.elevator_rl.elevator import Elevator
from src.elevator_rl.house import House
from src.elevator_rl.house import RequestDirection

# Configuration

FLOOR_HEIGHT = 100
ELEVATOR_WIDTH = ELEVATOR_HEIGHT = 100
ELEVATOR_SPACING = ELEVATOR_WIDTH + 50


def triangle_down(floor: int, color: str):
    pts_down = np.array([[0, 0], [20, 0], [10, -np.sqrt(25 ** 2 - 10 ** 2)]])
    pts_down += [-50, 45 + FLOOR_HEIGHT * floor]
    triangle = Polygon(pts_down, closed=True, color=color)
    plt.gca().add_patch(triangle)


def triangle_up(floor: int, color: str):
    pts_up = np.array([[0, 0], [20, 0], [10, np.sqrt(25 ** 2 - 10 ** 2)]])
    pts_up += [-50, 50 + FLOOR_HEIGHT * floor]
    triangle = Polygon(pts_up, closed=True, color=color)
    plt.gca().add_patch(triangle)


def draw_elevator(
    idx: int, floor: int, stop_requests: List[int], nr_of_passengers: int, height: int
):
    # elevator schacht
    rectangle = plt.Rectangle(
        (ELEVATOR_SPACING * idx, 0), ELEVATOR_WIDTH, height, ec="0.5", fc="0.5"
    )
    plt.gca().add_patch(rectangle)

    # position of elevator
    rectangle = plt.Rectangle(
        (ELEVATOR_SPACING * idx, floor * FLOOR_HEIGHT),
        ELEVATOR_WIDTH,
        ELEVATOR_HEIGHT,
        fc="b",
    )
    plt.gca().add_patch(rectangle)
    plt.text(
        idx * ELEVATOR_SPACING + 0.42 * ELEVATOR_WIDTH,
        floor * FLOOR_HEIGHT + 0.4 * ELEVATOR_HEIGHT,
        nr_of_passengers,
        color="w",
    )

    # stop_requests
    for floor in stop_requests:
        circle = plt.Circle(
            (
                ELEVATOR_SPACING * idx + 0.5 * ELEVATOR_WIDTH,
                (floor + 0.5) * FLOOR_HEIGHT,
            ),
            15,
            fc="w",
        )
        plt.gca().add_patch(circle)


def draw_passenger_request(
    floor: int, direction: RequestDirection
):  # TODO this has to indicate up/down as well
    if direction == RequestDirection.DOWN:
        triangle_down(floor, "r")
    else:
        triangle_up(floor, "g")


def render(house: House, elevators: List[Elevator]):
    width = len(elevators) * ELEVATOR_SPACING
    height = house.number_of_floors * FLOOR_HEIGHT
    plt.axes()
    plt.gcf().set_size_inches(10, 10)
    floors = list(range(0, house.number_of_floors * FLOOR_HEIGHT, FLOOR_HEIGHT))
    plt.hlines(floors, 0, width, colors="0.8", linestyles="dotted")
    plt.axis("scaled")
    plt.axis("off")
    plt.gca().set_ylim(0, height + 30)

    # Display floor numbers
    for i in range(0, house.number_of_floors):
        plt.text(-100, (i + 0.4) * FLOOR_HEIGHT, i)
        triangle_down(i, "0.9")
        triangle_up(i, "0.9")

    # Display elevator numbers
    for i in range(0, len(elevators)):
        plt.text(i * ELEVATOR_SPACING + 0.4 * ELEVATOR_WIDTH, -50, i)

    for i, elevator in enumerate(elevators):
        draw_elevator(
            i, elevator.floor, elevator.floor_requests, elevator.passengers, height
        )
    for request in house.elevator_requests:
        draw_passenger_request(request.floor, request.direction)

    plt.show()


def main():
    house = House(8, [])
    elevators = [Elevator(5, 2, 0, []), Elevator(5, 2, 0, [2, 3])]
    render(house, elevators)


if __name__ == "__main__":
    main()
