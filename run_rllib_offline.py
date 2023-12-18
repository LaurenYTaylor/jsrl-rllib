from ray.rllib.algorithms.pg import PGConfig
from ray.tune.logger import pretty_print

algo = (
    PGConfig()
    .environment(env="CartPole-v1")
    .framework("torch")
    .offline_data(output="offline_data/cartpole-out", output_max_file_size=5000000)
    .build())

for i in range(100):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")