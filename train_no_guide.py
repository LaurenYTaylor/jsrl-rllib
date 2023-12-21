from datetime import datetime

def train_no_guide(algo_config, env, timestr, num_iterations=300, checkpoint_freq=5, checkpoint_path=None, eval_interval=5, eval_duration=20):
    
    if checkpoint_path is None:
        checkpoint_path = f"models/{env}_{algo_config.__name__}_{timestr}/noguide"
        log_path = f"logs/{env}_{algo_config.__name__}_{timestr}/noguide"

    algo = (
        algo_config()
        .environment(env=env)
        .framework("torch")
        .evaluation(
            evaluation_num_workers=1,
            evaluation_duration_unit="episodes",
            evaluation_interval=eval_interval,
            evaluation_duration=eval_duration
        )
        .reporting(metrics_num_episodes_for_smoothing=1)
        .resources(num_learner_workers=1, num_cpus_per_worker=1)
        .debugging(logger_config={"logdir": log_path,
                                  "type": "ray.tune.logger.TBXLogger"})
    ).build()

    for it in range(num_iterations):
        print(f"No Guide Training: {it}/{num_iterations}")
        result = algo.train()
        
        if it % checkpoint_freq == 0:
            algo.save(checkpoint_dir=checkpoint_path)

if __name__ == "__main__":
    now = datetime.now()
    formatted = now.strftime("%m%d%Y_%H%M%S")
    
    from ray.rllib.algorithms.dqn import DQNConfig
    train_no_guide(DQNConfig, "CartPole-v1", formatted)