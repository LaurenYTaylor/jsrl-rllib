from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.cql import CQL
import numpy as np
from d4rl_env_maker import antmaze_umaze_render

path = "/home/a1899783/Documents/Code/jsrl-rllib/models/d4rl_env_maker.antmaze_umaze_CQLConfig_01042024_163356/offline"
env = antmaze_umaze_render(render_mode="human")
algo = CQL.from_checkpoint(path)

"""
data = []
for i in range(10):
    print(f"Eval: {i}/10")
    data.append(algo.evaluate()["evaluation"]["sampler_results"]["episode_reward_mean"])
"""
# np.save("data/guide_policy_res.npy", np.array(data))
episode_reward = 0
terminated = truncated = False
obs, info = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

print(reward)
