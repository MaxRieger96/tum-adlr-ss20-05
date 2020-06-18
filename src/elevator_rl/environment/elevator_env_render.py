import os
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from elevator_rl.environment.elevator import ElevatorEnvAction
from matplotlib.patches import Polygon

from elevator_rl.environment.house import House

# TODO center all texts

# configuration
FLOOR_HEIGHT = 100
ELEVATOR_WIDTH = ELEVATOR_HEIGHT = 100
ELEVATOR_SPACING = ELEVATOR_WIDTH + 50


class RequestDirection(Enum):
    DOWN = -1
    UP = 1


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


def draw_rectangle(x: float, y: float, color: str):
    pts = np.array([[-10, -10], [-10, 10], [10, 10], [10, -10]], dtype=float)
    pts += np.array([x, y])
    square = Polygon(pts, closed=True, color=color)
    plt.gca().add_patch(square)


def draw_elevator(
        idx: int, floor: int, stop_requests: List[int], nr_of_passengers: int, height: int
):
    # elevator shaft
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
    # load of elevator
    plt.text(
        idx * ELEVATOR_SPACING + 0.42 * ELEVATOR_WIDTH,
        floor * FLOOR_HEIGHT + 0.4 * ELEVATOR_HEIGHT,
        nr_of_passengers,
        fontdict={'weight': 'bold'},
        color="#ffffff",
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


def draw_passenger_request(floor: int, direction: RequestDirection, time: float, method: str):
    if direction == RequestDirection.DOWN:
        triangle_down(floor, "r")
        if method != "file":
            plt.text(-50, 45 + FLOOR_HEIGHT * floor, f"{time:.1f}")  # TODO better text pos
    else:
        triangle_up(floor, "g")
        if method != "file":
            plt.text(-50, 50 + FLOOR_HEIGHT * floor, f"{time:.1f}")  # TODO better text pos


def render(house: House, method: str, prev_time: float, action: ElevatorEnvAction):
    method = method if method in ["matplotlib", "file"] else "matplotlib"
    width = len(house.elevators) * ELEVATOR_SPACING
    height = house.number_of_floors * FLOOR_HEIGHT
    font = {'size': 22}
    plt.rc('font', **font)
    plt.axes()
    plt.gcf().set_size_inches(10, 10)
    floors = list(range(0, house.number_of_floors * FLOOR_HEIGHT, FLOOR_HEIGHT))
    plt.hlines(floors, 0, width, colors="0.8", linestyles="dotted")
    plt.axis("scaled")
    plt.axis("off")
    plt.gca().set_ylim(-100, height + 30)
    plt.gca().set_xlim(
        -100, len(house.elevators) * ELEVATOR_SPACING
    )  # TODO set better limits (-50 for max?)

    # display floor numbers
    for i in range(0, house.number_of_floors):
        plt.text(-100, (i + 0.4) * FLOOR_HEIGHT, i)
        triangle_down(i, "0.9")
        triangle_up(i, "0.9")

    # highlight next elevator
    if method != "file":
        draw_rectangle(
            house.next_to_move() * ELEVATOR_SPACING + 0.5 * ELEVATOR_WIDTH, -25, "#ff9900"
        )

    # Display action took
    if action:
        plt.text(action.elevator_idx * ELEVATOR_SPACING + 0.5 * ELEVATOR_WIDTH,
                 -50,
                 action.elevator_action.name,
                 fontdict={'weight': 'bold'},
                 horizontalalignment="center",
                 verticalalignment="center")

    # display elevator numbers and times
    for i in range(0, len(house.elevators)):
        plt.text(
            i * ELEVATOR_SPACING + 0.5 * ELEVATOR_WIDTH,
            -25,
            f"Elev. #{i}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        if method != "file":
            plt.text(
                i * ELEVATOR_SPACING + 0.5 * ELEVATOR_WIDTH,
                -70,
                f"Elev. Time: {house.elevators[i].time:.0f}",
                horizontalalignment="center",
                verticalalignment="center",
            )

    # display total time
    plt.text(
        ELEVATOR_SPACING * len(house.elevators) / 2,
        -100,
        f"Total Time: {house.time:.0f}",
        horizontalalignment="center",
        verticalalignment="center",
    )

    for i, elevator in enumerate(house.elevators):
        stop_requests = list(np.where(elevator.floor_requests)[0])
        draw_elevator(
            idx=i,
            floor=elevator.floor,
            stop_requests=stop_requests,
            nr_of_passengers=len(elevator.passengers),
            height=height,
        )

    for i, request in enumerate(house.up_requests):
        if request:
            draw_passenger_request(
                floor=i,
                direction=RequestDirection.UP,
                time=house.up_requests_waiting_since[i],
                method=method,
            )
    for i, request in enumerate(house.down_requests):
        if request:
            draw_passenger_request(
                floor=i,
                direction=RequestDirection.DOWN,
                time=house.down_requests_waiting_since[i],
                method=method,
            )
    if method == "matplotlib":
        plt.show()
    else:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        start_time = int(prev_time)
        stop_time = int(house.time)
        for time in range(start_time, stop_time):
            plt.savefig("{}/step_{}.png".format(os.path.join(ROOT_DIR, "../plots"), time))
        plt.show()
        plt.close()
