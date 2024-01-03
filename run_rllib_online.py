from custom_algorithm import make_custom_algorithm
from custom_callbacks import UpdateThresholdCallback
from horizon_fns import timestep_horizon
from data_reader_utils import json_input_creator, d4rl_input_creator


def train_learning_policy(
    algo,
    algo_config,
    guide_alg,
    env,
    timestr,
    offline_data_path,
    guide_policy_path,
    deterministic_sample,
    num_iterations=300,
    checkpoint_freq=5,
    checkpoint_path=None,
    eval_interval=5,
    eval_duration=10,
):
    """_summary_

    Args:
        algo (_type_): _description_
        algo_config (_type_): _description_
        guide_alg (_type_): _description_
        env (_type_): _description_
        timestr (_type_): _description_
        offline_data_path (_type_): _description_
        guide_policy_path (_type_): _description_
        deterministic_sample (_type_): _description_
        num_iterations (int, optional): _description_. Defaults to 100.
        checkpoint_freq (int, optional): _description_. Defaults to 5.
        checkpoint_path (_type_, optional): _description_. Defaults to None.
        eval_interval (int, optional): _description_. Defaults to 50.
        eval_duration (int, optional): _description_. Defaults to 20.
    """
    if checkpoint_path is None:
        checkpoint_path = f"models/{env}_{algo_config.__name__}_{timestr}/online"
        log_path = f"logs/{env}_{algo_config.__name__}_{timestr}/online"

    custom_algo = make_custom_algorithm(algo)
    algo_config = algo_config(algo_class=custom_algo)

    if offline_data_path is not None:
        data_reader = json_input_creator
        input_config = {"input_files": offline_data_path}
    else:
        data_reader = d4rl_input_creator
        input_config = {"inputs": env}

    config = (
        algo_config.environment(env=env)
        .framework("torch")
        .offline_data(
            input_={"sampler": 0.75, data_reader: 0.25},
            input_config=input_config,
            postprocess_inputs=True,
        )
        .evaluation(
            evaluation_num_workers=1,
            evaluation_duration_unit="episodes",
            evaluation_interval=eval_interval,
            evaluation_duration=eval_duration,
            evaluation_config={"input": "sampler"},
        )
        .reporting(metrics_num_episodes_for_smoothing=1)
        .callbacks(callbacks_class=UpdateThresholdCallback)
        .resources(num_learner_workers=1, num_cpus_per_worker=1)
        .debugging(
            logger_config={"logdir": log_path, "type": "ray.tune.logger.TBXLogger"}
        )
    )

    config = config.update_from_dict(
        {
            "jsrl": {
                "guide_policy": (guide_alg, guide_policy_path),
                "deterministic_sample": deterministic_sample,
                "max_horizon": 500,
                "curriculum_stages": 10,
                "horizon_fn": timestep_horizon,
                "rolling_mean_n": 5,
                "tolerance": 0.01,
            }
        }
    )
    algo = config.build()

    for it in range(num_iterations):
        print(f"Online Training: {it}/{num_iterations}")
        algo.train()

        if it % checkpoint_freq == 0:
            algo.save(checkpoint_dir=checkpoint_path)
