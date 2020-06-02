import numpy as np

from elevator_rl.house import House


def get_10_story_house() -> House:
    elevator_capacity = 20
    number_of_elevators = 4
    number_of_floors = 10

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


def get_simple_house() -> House:
    elevator_capacity = 10
    number_of_elevators = 1
    number_of_floors = 3

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
