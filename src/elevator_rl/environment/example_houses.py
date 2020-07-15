import numpy as np

from elevator_rl.environment.house import House


def get_10_story_house() -> House:
    return produce_house(20, 4, 10)


def get_8_story_house() -> House:
    return produce_house(20, 2, 8)


def get_5_story_house() -> House:
    return produce_house(10, 2, 5)


def get_simple_house() -> House:
    return produce_house(10, 1, 3)


def get_10_story_single_elev_house() -> House:
    return produce_house(20, 1, 10)


def produce_house(
        elevator_capacity: int, number_of_elevators: int, number_of_floors: int
) -> House:
    request_rates = np.ones(number_of_floors)
    request_rates /= 60
    request_rates[0] *= 2

    target_probabilities = np.ones(number_of_floors)
    target_probabilities[0] *= 2
    target_probabilities /= np.sum(target_probabilities)

    house = House(
        number_of_floors,
        elevator_capacity,
        number_of_elevators,
        request_rates,
        target_probabilities,
    )
    return house
