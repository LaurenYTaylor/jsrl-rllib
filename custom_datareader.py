from ray.rllib.offline import D4RLReader, IOContext, JsonReader
from ray.rllib.utils.annotations import override
from copy import deepcopy

from ray.rllib.policy.sample_batch import (
    SampleBatch,
    concat_samples,
    convert_ma_batch_to_sample_batch,
)
from ray.rllib.utils.annotations import override

from ray.rllib.utils.typing import SampleBatchType, Dict
import d4rl_env_maker
import numpy as np


class CustomJsonReader(JsonReader):
    """
    Custom JSON reader to accept offline data from specified input files.
    """

    def __init__(self, ioctx: IOContext):
        super().__init__(ioctx.input_config["input_files"], ioctx)

    @override(JsonReader)
    def _postprocess_if_needed(self, batch: SampleBatchType) -> SampleBatchType:
        """Postprocess that deletes keys from the offline batches, if those keys are
        not in the view requirements. This can cause problems when you use different
        algorithms for offline training and online refinement.

        Args:
            batch (SampleBatchType): An offline batch.

        Returns:
            SampleBatchType: The processed offline batch.
        """
        if not self.ioctx.config.get("postprocess_inputs"):
            return batch

        batch = convert_ma_batch_to_sample_batch(batch)

        if isinstance(batch, SampleBatch):
            out = []
            for sub_batch in batch.split_by_episode():
                postprocessed = self.default_policy.postprocess_trajectory(sub_batch)
                postprocessed_copy = deepcopy(postprocessed)
                for key in postprocessed.keys():
                    if key not in self.default_policy.view_requirements:
                        del postprocessed_copy[key]
                out.append(postprocessed_copy)
            return concat_samples(out)
        else:
            # TODO(ekl) this is trickier since the alignments between agent
            #  trajectories in the episode are not available any more.
            raise NotImplementedError(
                "Postprocessing of multi-agent data not implemented yet."
            )


class CustomD4RLReader(D4RLReader):
    """A custom D4RL data reader. This is needed for D4RL environments that have not yet been
    migrated to gymnasium, such as AntMaze.

    Args:
        D4RLReader (_type_): _description_
    """

    @override(D4RLReader)
    def __init__(self, inputs: str, batch_size, ioctx: IOContext = None):
        """Initializes a CustomD4RLReader instance.

        Args:
            inputs: String corresponding to the D4RL environment creation function.
            ioctx: Current IO context object.
        """
        import d4rl

        env_name = inputs.split(".")[1]
        env_fn = getattr(d4rl_env_maker, env_name)
        self.env = env_fn().env
        self.dataset = _convert_to_batch(qlearning_dataset(self.env, batch_size))
        assert self.dataset.count >= 1
        self.counter = 0


def qlearning_dataset(env, batch_size, dataset=None, terminate_on_end=True, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    terminated_ = []
    truncated_ = []
    t_ = []
    eps_id_ = []
    agent_index_ = []
    rollout_id_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    ep_id = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            ep_id += 1
            continue
        if done_bool or final_timestep:
            episode_step = 0
            ep_id += 1

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        terminated_.append(done_bool)
        truncated_.append(final_timestep)
        t_.append(episode_step)
        eps_id_.append(ep_id)
        agent_index_.append(0)
        rollout_id_.append(int(episode_step / batch_size))
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(terminated_),
        "truncateds": np.array(truncated_),
        "ts": np.array(t_),
        "eps_ids": np.array(eps_id_),
        "agent_index": np.array(agent_index_),
        "rollout_ids": np.array(rollout_id_),
    }


def _convert_to_batch(dataset: Dict) -> SampleBatchType:
    # Converts D4RL dataset to SampleBatch
    d = {}
    d[SampleBatch.OBS] = dataset["observations"]
    d[SampleBatch.ACTIONS] = dataset["actions"]
    d[SampleBatch.NEXT_OBS] = dataset["next_observations"]
    d[SampleBatch.REWARDS] = dataset["rewards"]
    d[SampleBatch.TERMINATEDS] = dataset["terminals"]
    d[SampleBatch.TRUNCATEDS] = dataset["truncateds"]
    d[SampleBatch.INFOS] = np.array([{}] * len(dataset["terminals"]))
    d[SampleBatch.T] = dataset["ts"]
    d[SampleBatch.EPS_ID] = dataset["eps_ids"]
    d[SampleBatch.AGENT_INDEX] = dataset["agent_index"]
    d[SampleBatch.UNROLL_ID] = dataset["rollout_ids"]

    return SampleBatch(d)
