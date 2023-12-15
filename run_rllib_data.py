from ray.rllib.algorithms.dqn import DQN, DQNConfig
from custom_algorithm import make_custom_algorithm
from custom_callbacks import UpdateThresholdCallback
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
import os

config = (
    DQNConfig()
    .environment(env="CartPole-v1")
    .framework("torch")
    .offline_data(input_="offline_data/cartpole-out")
    .evaluation(
        evaluation_interval=100,
        evaluation_duration=100,
        evaluation_num_workers=1,
        evaluation_duration_unit="timesteps"
        #evaluation_config={"input": "offline_data/cartpole-out"},
    )
    .reporting(metrics_num_episodes_for_smoothing=1)
    .callbacks(callbacks_class=UpdateThresholdCallback)
)


algo = config.build()
for it in range(100):
    result = algo.train()

    if it%10 == 0:
        print(pretty_print(result))
        algo.save(checkpoint_dir="offline_data/partially_trained")