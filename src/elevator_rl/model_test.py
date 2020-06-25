from elevator_rl.alphazero.model import NNModel
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.example_houses import produce_house


def main():
    house = produce_house(
        elevator_capacity=10, number_of_elevators=3, number_of_floors=7
    )
    env = ElevatorEnv(house)
    sample_observation = env.get_observation().as_array()
    house_obs_dims = sample_observation[0].shape[0]
    elevator_obs_dims = sample_observation[1].shape[0]
    model = NNModel(
        house_observation_dims=house_obs_dims,
        elevator_observation_dims=elevator_obs_dims,
        policy_dims=3,
    )
    model.eval()
    policy, value = model.get_policy_and_value(env)
    print(policy)
    print(value)


if __name__ == "__main__":
    main()
