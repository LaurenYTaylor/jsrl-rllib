import os
os.environ["RAY_DEDUP_LOGS"] = "0"
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from custom_algorithm import make_custom_algorithm
from custom_callbacks import UpdateThresholdCallback
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from horizon_fns import timestep_horizon

def train_learning_policy(algo, algo_config, guide_alg, env, timestr, offline_data_path, guide_policy_path, deterministic_sample, 
                          num_iterations=100, checkpoint_freq=5, checkpoint_path=None, eval_interval=50, eval_duration=20):
    if checkpoint_path is None:
        checkpoint_path = f"models/{env}_{algo_config}_{timestr}/online"
        log_path = f"logs/{env}_{algo_config}_{timestr}/online"
        
    custom_algo = make_custom_algorithm(algo)
    algo_config = algo_config(algo_class=custom_algo)

    config = (
        algo_config
        .environment(env=env)
        .framework("torch")
        #.offline_data(input_={"sampler": 0.75, "offline_data/cartpole-out": 0.25})
        .evaluation(
            evaluation_num_workers=1,
            evaluation_duration_unit="episodes"
        )
        .reporting(metrics_num_episodes_for_smoothing=1)
        .callbacks(callbacks_class=UpdateThresholdCallback)
        .resources(num_learner_workers=1)
        .debugging(logger_config={"logdir": log_path,
                                  "type": "ray.tune.logger.TBXLogger"})
    )

    config = config.update_from_dict({"jsrl": {"guide_policy": (guide_alg, guide_policy_path),
                                            "deterministic_sample": deterministic_sample,
                                            "max_horizon": 5,
                                            "curriculum_stages": 10,
                                            "horizon_fn": timestep_horizon,
                                            "rolling_mean_n": 5,
                                            "tolerance": 0.01}})
    algo = config.build()

    for it in range(num_iterations):
        result = algo.train()
        
        if it % checkpoint_freq == 0:
            algo.save(checkpoint_dir=checkpoint_path)
        
        if it % eval_interval == 0:
            algo.evaluate(lambda x: x < eval_duration)