Observation Sensitive MCTS for Elevator Transportation
======================================================

In this project, we try to adapt the AlphaZero algorithm to problems with continuous
rewards. We chose the problem of elevator transportation to apply our version of the
algorithm.

Project Structure
-----------------

- More information about the Problem can be found in the [Final Report](./Final_Report.pdf)
- The main training procedure can be found in [train.py](./src/elevator_rl/train.py)
- Hyper-parameters can be set at [config.yaml](./src/elevator_rl/config.yaml),
use the environment variable "CONFIG_NAME" to choose a configuration
- The implementation of the model and learning algorithm can be found in the
[alphazero folder](./src/elevator_rl/alphazero)
- A detailed description of the environment we used can be found in the
[Documentation](./doc/environment.md), the implementation is in the
[environment folder](./src/elevator_rl/environment)
- In the [baseline folder](./src/elevator_rl/baseline) you can find baselines such as
random policy, pure MCTS, and the heuristic collective control
- If you want to play a little bit and control some elevators yourself you can run the
[interactive environment](./src/elevator_rl/environment/interactive_env.py)
