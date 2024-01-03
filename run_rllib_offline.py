from data_reader_utils import d4rl_input_creator


def train_guide_policy(
    algo_config,
    env,
    timestr,
    offline_data_path,
    num_iterations=300,
    checkpoint_freq=20,
    checkpoint_path=None,
    eval_interval=5,
    eval_duration=20,
):
    if checkpoint_path is None:
        checkpoint_path = f"models/{env}_{algo_config.__name__}_{timestr}/offline"
        log_path = f"logs/{env}_{algo_config.__name__}_{timestr}/offline"

    algo = (
        algo_config()
        .environment(env=env, disable_env_checking=True)
        .framework("torch")
        .offline_data(input_=d4rl_input_creator, input_config={"env": env})
        .evaluation(
            evaluation_num_workers=1,
            evaluation_duration_unit="episodes",
            evaluation_config={"input": "sampler"},
            evaluation_interval=eval_interval,
            evaluation_duration=eval_duration,
        )
        .resources(num_learner_workers=1, num_cpus_per_worker=1)
        .debugging(
            logger_config={"logdir": log_path, "type": "ray.tune.logger.TBXLogger"}
        )
        .build()
    )

    for it in range(num_iterations):
        print(f"Offline Training: {it}/{num_iterations}")
        algo.train()
        if it % checkpoint_freq == 0 or it == num_iterations - 1:
            algo.save(checkpoint_dir=checkpoint_path)

    return checkpoint_path
