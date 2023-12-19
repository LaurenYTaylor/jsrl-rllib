from datetime import datetime
import argparse
from ray.rllib.algorithms.pg import PGConfig, PG
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from run_rllib_make_data import make_offline_dataset
from run_rllib_offline import train_guide_policy
from run_rllib_online import train_learning_policy

ALGO_DICT = {"pg": PG,
            "dqn": DQN,
            "ppo": PPO}
ALGO_CONFIG_DICT = {"pg": PGConfig,
            "dqn": DQNConfig,
            "ppo": PPOConfig}

def run_pipeline(env, data_algo_config, offline_algo, offline_algo_config, online_algo, online_algo_config):
    now = datetime.now()
    formatted = now.strftime("%m%d%Y_%H%M%S")
    
    offline_data_path = make_offline_dataset(data_algo_config, env, formatted)
    guide_checkpoint_path = train_guide_policy(offline_algo_config, env, formatted, offline_data_path)
    
    if online_algo in [PPO, PG]:
        deterministic_sample = False
    else:
        deterministic_sample = True
        
    train_learning_policy(online_algo, online_algo_config, offline_algo, env, formatted, offline_data_path, guide_checkpoint_path, deterministic_sample)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--data_creation_algo", "-dat", type=str, default="pg")
    parser.add_argument("--offline_train_algo", "-off", type=str, default="dqn")
    parser.add_argument("--online_train_algo", "-on", type=str, default="dqn")
    args = parser.parse_args()
    
    run_pipeline(args.env,
                 ALGO_CONFIG_DICT[args.data_creation_algo],
                 ALGO_DICT[args.offline_train_algo],
                 ALGO_CONFIG_DICT[args.offline_train_algo],
                 ALGO_DICT[args.online_train_algo],
                 ALGO_CONFIG_DICT[args.online_train_algo])