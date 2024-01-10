def make_offline_dataset(
    algo_config,
    env,
    timestr,
    n_data=1000000,
    num_iterations=100,
    checkpoint_freq=5,
    checkpoint_path=None,
):
    """Makes an offline dataset for training an agent offline.

    Args:
        algo_config (AlgorithmConfig): RLLib algorithm configuration.
        env (gymnasium.Env): Any env that runs with RLLib.
        timestr (str): Current date time.
        n_data (int, optional): Desired number of samples in offline dataset. Defaults to 1000000.
        num_iterations (int, optional): Number of training iterations. Defaults to 100.
        checkpoint_freq (int, optional): Model checkpointing frequency. Defaults to 5.
        checkpoint_path (_type_, optional): Path to checkpoint. Defaults to None, checkpoint is created from env_alg_timestr.
    """
    if checkpoint_path is None:
        checkpoint_path = f"models/{env}_{algo_config.__name__}_{timestr}/data"

    offline_dataset_path = f"offline_data/{env}_{algo_config.__name__}_{timestr}"

    algo = (
        algo_config()
        .environment(env=env)
        .framework("torch")
        .offline_data(output=offline_dataset_path, output_max_file_size=n_data)
        .resources(num_cpus_per_worker=1)
        .rollouts(num_rollout_workers=1, num_envs_per_worker=1)
        .build()
    )

    for it in range(num_iterations):
        print(f"Data Creation: {it}/{num_iterations}")
        algo.train()
        if it % checkpoint_freq == 0:
            algo.save(checkpoint_path)
    return offline_dataset_path
