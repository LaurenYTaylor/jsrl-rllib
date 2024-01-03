from ray.rllib.algorithms.dqn import DQN
import numpy as np

path = "/home/a1899783/Documents/Code/jsrl-rllib/models/CartPole-v1_DQNConfig_12202023_125957/offline"
algo = DQN.from_checkpoint(path)

data = []
for i in range(10):
    print(f"Eval: {i}/10")
    data.append(algo.evaluate()["evaluation"]["sampler_results"]["episode_reward_mean"])

np.save("data/guide_policy_res.npy", np.array(data))
