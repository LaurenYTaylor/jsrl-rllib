from custom_algorithm import make_custom_algorithm
from custom_callbacks import UpdateThresholdCallback
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
import os


def train_guide_policy(algo_config, env, timestr, offline_data_path, num_iterations=100, checkpoint_freq=5,
                       checkpoint_path=None, eval_interval=50, eval_duration=20):
    if checkpoint_path is None:
        checkpoint_path = f"models/{env}_{algo_config}_{timestr}/offline"
    
    algo = (
        algo_config()
        .environment(env=env)
        .framework("torch")
        .offline_data(input_=offline_data_path)
        .evaluation(
            evaluation_num_workers=1,
            evaluation_duration_unit="episodes",
            evaluation_config={"input": "sampler"}
        )
        .build()
    )
    
    for it in range(num_iterations):
        result = algo.train()

        if it % checkpoint_freq == 0:
            algo.save(checkpoint_dir=checkpoint_path)
        
        if it % eval_interval == 0:
            algo.evaluate(lambda x: x < eval_duration)
            
    return checkpoint_path