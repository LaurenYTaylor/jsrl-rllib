import os
os.environ["RAY_DEDUP_LOGS"] = "0"
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from custom_algorithm import make_custom_algorithm
from custom_callbacks import UpdateThresholdCallback
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from horizon_fns import timestep_horizon

#print(dir(DQNConfig))
#import pdb;pdb.set_trace()
#exit()

algo = DQN
custom_algo = make_custom_algorithm(algo)
algo_config = DQNConfig(algo_class=custom_algo)

path = "offline_data/partially_trained"

config = (
    algo_config
    .environment(env="CartPole-v1")
    .framework("torch")
    #.offline_data(input_="offline_data/cartpole-out")
    .evaluation(
        evaluation_duration=20,
        evaluation_num_workers=1,
        evaluation_duration_unit="episodes"
        #evaluation_config={"input": "offline_data/cartpole-out"},
    )
    .reporting(metrics_num_episodes_for_smoothing=1)
    .callbacks(callbacks_class=UpdateThresholdCallback)
    .resources(num_learner_workers=0)
)

if algo in [PPO]:
    deterministic_sample = False
else:
    deterministic_sample = True

config = config.update_from_dict({"jsrl": {"guide_policy": (DQN, path),
                                           "deterministic_sample": deterministic_sample,
                                           "max_horizon": 500,
                                           "curriculum_stages": 10,
                                           "horizon_fn": timestep_horizon
                                           "rolling_mean_n": 5,
                                           "tolerance": 0.01}})
algo = config.build()

for it in range(30):
    result = algo.train()
    
    if it%10 == 0:
        #print(pretty_print(result))
        algo.save(checkpoint_dir="offline_data/")
    if it%5 == 0:
        algo.evaluate()