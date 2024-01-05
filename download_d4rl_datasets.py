import gym
import d4rl
import numpy as np

env_name = "antmaze-umaze-v2"

env = gym.make(env_name)

dataset = env.get_dataset()
print(dataset["actions"])

# d4rl.qlearning_dataset also adds next_observations.
dataset = d4rl.qlearning_dataset(env)
print(dataset["actions"])
print(np.where(dataset["actions"] < -1))
print(np.where(dataset["actions"] > 1))
