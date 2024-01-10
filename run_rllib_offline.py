from data_reader_utils import d4rl_input_creator
import d4rl_env_maker
import numpy as np
from ray.rllib.policy.policy import Policy
import gymnasium


def train_guide_policy(
    algo_config,
    training_algo_config,
    env,
    timestr,
    offline_data_path,
    num_iterations=300,
    checkpoint_freq=20,
    checkpoint_path=None,
    eval_interval=10,
    eval_duration=10,
    train_batch_size=256,
):
    if checkpoint_path is None:
        checkpoint_path = f"models/{env}_{algo_config.__name__}_{timestr}/offline"
        log_path = f"logs/{env}_{algo_config.__name__}_{timestr}/offline"

    input_config = {"env": env}

    if offline_data_path is None:
        input = d4rl_input_creator
        input_config["batch_size"] = train_batch_size
    else:
        input = offline_data_path

    algo = (
        algo_config()
        .environment(env=env, disable_env_checking=True)
        .framework("torch")
        .training(train_batch_size=train_batch_size, **training_algo_config["training"])
        .offline_data(
            input_=input,
            input_config=input_config,
            offline_sampling=True,
            **training_algo_config["offline_data"],
        )
        .evaluation(
            evaluation_num_workers=1,
            evaluation_duration_unit="episodes",
            evaluation_config={"input": "sampler"},
            evaluation_interval=eval_interval,
            evaluation_duration=eval_duration,
        )
        .resources(num_cpus_per_worker=1)
        .debugging(
            logger_config={"logdir": log_path, "type": "ray.tune.logger.TBXLogger"},
            seed=42,
        )
        .rollouts(
            **training_algo_config["rollouts"],
            num_rollout_workers=4,
            num_envs_per_worker=2,
            create_env_on_local_worker=False,
        )
        .build()
    )

    for it in range(num_iterations):
        print(f"Offline Training: {it}/{num_iterations}")
        algo.train()
        if it % checkpoint_freq == 0 or it == num_iterations - 1:
            algo.save(checkpoint_dir=checkpoint_path)

    return checkpoint_path


def evaluate_guide_policy(
    checkpoint_path, env, algo, horizon_fn, horizon_accumulator_fn, eval_eps=5
):
    try:
        env_name = env.split(".")[1]
        env_fn = getattr(d4rl_env_maker, env_name)
        env = env_fn()
    except IndexError:
        env_name = env
        env = gymnasium.make(env)

    algo = Policy.from_checkpoint(checkpoint_path + "/policies/default_policy")

    horizons = []
    for ep in range(eval_eps):
        episode_reward = 0
        terminated = truncated = False
        obs, info = env.reset()
        action = None
        t = 1
        step_thresholds = []
        while not terminated and not truncated:
            step_threshold = horizon_fn(t, {"obs": obs, "action": action})
            step_thresholds.append(step_threshold)
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action[0])
            episode_reward += reward
            t += 1
        horizons.append(horizon_accumulator_fn(step_thresholds))
    return np.mean(horizons)
