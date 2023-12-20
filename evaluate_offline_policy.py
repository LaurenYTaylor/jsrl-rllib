from ray.rllib.algorithms.dqn import DQN

path = "/home/a1899783/Documents/Code/jsrl-rllib/models/CartPole-v1_DQNConfig_12202023_125957/offline"
algo = DQN.from_checkpoint(path)

print(algo.evaluate())



