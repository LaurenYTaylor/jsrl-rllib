import numpy as np


def timestep_horizon(episode_length, _):
    return episode_length


def antmaze_goal_horizon(episode_length, input_dict):
    import pdb

    pdb.set_trace()


def max_horizon_fn(horizons):
    return np.max(horizons)


def mean_horizon_fn(horizons):
    return np.mean(horizons)
