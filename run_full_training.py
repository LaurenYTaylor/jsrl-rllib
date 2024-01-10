import os

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
import ray
from datetime import datetime
import argparse
from ray.rllib.algorithms.pg import PGConfig, PG
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.sac import SACConfig, SAC
from ray.rllib.algorithms.bc import BCConfig, BC
from ray.rllib.algorithms.cql import CQLConfig, CQL
from horizon_fns import timestep_horizon, mean_horizon_fn, max_horizon_fn
from run_rllib_make_data import make_offline_dataset
from run_rllib_offline import train_guide_policy, evaluate_guide_policy
from run_rllib_online import train_learning_policy
from training_configs import training_configs

ALGO_DICT = {"pg": PG, "dqn": DQN, "ppo": PPO, "sac": SAC, "bc": BC, "cql": CQL}
ALGO_CONFIG_DICT = {
    "pg": PGConfig,
    "dqn": DQNConfig,
    "ppo": PPOConfig,
    "sac": SACConfig,
    "bc": BCConfig,
    "cql": CQLConfig,
}


def run_pipeline(
    env,
    data_creation_algo,
    offline_train_algo,
    online_train_algo,
    offline_data_path=None,
    guide_checkpoint_path=None,
):
    data_algo_config = ALGO_CONFIG_DICT[data_creation_algo]
    offline_algo = ALGO_DICT[offline_train_algo]
    offline_algo_config = ALGO_CONFIG_DICT[offline_train_algo]
    online_algo = ALGO_DICT[online_train_algo]
    online_algo_config = ALGO_CONFIG_DICT[online_train_algo]

    now = datetime.now()
    formatted = now.strftime("%m%d%Y_%H%M%S")

    if offline_data_path is None and "d4rl" not in env:
        print("Making Offline Dataset")
        offline_data_path = make_offline_dataset(data_algo_config, env, formatted)
        ray.shutdown()

    if guide_checkpoint_path is None:
        print("Training Guide Policy")
        offline_training_config = training_configs[offline_train_algo]
        guide_checkpoint_path = train_guide_policy(
            offline_algo_config,
            offline_training_config,
            env,
            formatted,
            offline_data_path,
            num_iterations=5000,
            checkpoint_freq=25,
            eval_interval=25,
            eval_duration=10,
        )
        ray.shutdown()

    init_horizon = evaluate_guide_policy(
        guide_checkpoint_path, env, offline_algo, timestep_horizon, max_horizon_fn
    )
    print(init_horizon)
    ray.shutdown()

    if online_algo in [PPO, PG, SAC]:
        deterministic_sample = False
    else:
        deterministic_sample = True

    print("Training Learning Policy")
    train_learning_policy(
        online_algo,
        online_algo_config,
        offline_algo,
        env,
        formatted,
        offline_data_path,
        guide_checkpoint_path,
        deterministic_sample,
        init_horizon,
        num_iterations=5000,
        eval_duration=10,
        eval_interval=25,
        checkpoint_freq=25,
    )

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--data_creation_algo", "-dat", type=str, default="pg")
    parser.add_argument("--offline_train_algo", "-off", type=str, default="dqn")
    parser.add_argument("--online_train_algo", "-on", type=str, default="dqn")
    parser.add_argument("--offline_data_path", "-off_dat", type=str, default=None)
    parser.add_argument("--trained_guide_model_path", "-guide", type=str, default=None)
    args = parser.parse_args()

    run_pipeline(
        args.env,
        args.data_creation_algo,
        args.offline_train_algo,
        args.online_train_algo,
        args.offline_data_path,
        args.trained_guide_model_path,
    )
