import gym
import d4rl

env_name = "antmaze-umaze-v2"

env = gym.make(env_name)

dataset = env.get_dataset()
print(dataset)

# d4rl.qlearning_dataset also adds next_observations.
dataset = d4rl.qlearning_dataset(env)

dataset.save(f"data/{env_name}")
