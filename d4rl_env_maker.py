import gym as old_gym
from maze_env_wrapper import EnvCompatibility
import gymnasium as gym

try:
    import d4rl

    d4rl.__name__  # Fool LINTer.
except ImportError:
    d4rl = None


def antmaze_umaze():
    return EnvCompatibility(old_gym.make("antmaze-umaze-v0"))


def halfcheetah_random():
    return gym.make("halfcheetah-random-v0")


def halfcheetah_medium():
    return gym.make("halfcheetah-medium-v0")


def halfcheetah_expert():
    return gym.make("halfcheetah-expert-v0")


def halfcheetah_medium_replay():
    return gym.make("halfcheetah-medium-replay-v0")
