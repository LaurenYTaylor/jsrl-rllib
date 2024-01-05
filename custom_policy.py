import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import ray

from ray.rllib.utils.annotations import (
    override,
)
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, PolicyState
from ray.rllib.utils.checkpoints import (
    CHECKPOINT_VERSION,
    try_import_msgpack,
)
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, deprecation_warning

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)


def make_custom_policy(algo):
    """Makes a custom policy that subclasses the default policy of the chosen algorithm.

    Args:
        algo (ray.rllib.algorithm): The chosen algorithm.

    Returns:
        ray.rllib.policy: The custom policy.
    """
    policy = algo.get_default_policy_class(algo.get_default_config())

    class JSRLPolicy(policy):
        @override(policy)
        def compute_actions_from_input_dict(
            self,
            input_dict: Dict[str, TensorType],
            explore: bool = None,
            timestep: Optional[int] = None,
            **kwargs,
        ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
            """Chooses an action from either the guide or learning policy, depending on if
            the current time step or goal is below the current horizon.

            Args:
                input_dict (Dict[str, TensorType]): dict containing observations.
                explore (bool, optional): Whether agent can explore (i.e. if not evaluating). Defaults to None.
                timestep (Optional[int], optional): CUrrent timestep. Defaults to None.

            Returns:
                Tuple[TensorType, List[TensorType], Dict[str, TensorType]]: The chosen guide or learning action.
            """
            is_evaluating = False

            action = super().compute_actions_from_input_dict(
                input_dict, explore, timestep, **kwargs
            )

            if "agent_type" in self.config["jsrl"]:
                is_evaluating = True

            if is_evaluating and kwargs["episodes"][-1].length == 0:
                if hasattr(self, "episode_step_thresholds"):
                    self.config["jsrl"]["mean_threshold"].append(
                        self.config["jsrl"]["horizon_accumulate_fn"](
                            self.episode_step_thresholds
                        )
                    )
                self.episode_step_thresholds = []

            if "episodes" in kwargs:
                step_threshold = self.config["jsrl"]["horizon_fn"](
                    kwargs["episodes"][-1].length, input_dict
                )

            # Decide to use learning or guide agent respectively
            if (
                "current_horizon" not in self.config["jsrl"]
                or step_threshold > self.config["jsrl"]["current_horizon"]
            ):
                # During evaluation, keep track of agent_type
                if is_evaluating:
                    try:
                        self.episode_step_thresholds.append(step_threshold)
                    except UnboundLocalError:
                        pass
                    self.config["jsrl"]["agent_type"].append(1)
                return action
            else:
                if is_evaluating:
                    try:
                        self.episode_step_thresholds.append(step_threshold)
                    except UnboundLocalError:
                        pass
                    self.config["jsrl"]["agent_type"].append(0)
                if "actions" in input_dict:
                    input_dict.pop("actions")

                # The following makes a guide action computed from the pre-trained policy have the same information as a learning action
                guide_action = np.array(
                    [
                        self.config["jsrl"]["guide_policy"].compute_single_action(
                            input_dict=input_dict, explore=False, timestep=timestep
                        )[0]
                    ]
                )

                act_np = np.array(guide_action)
                if self.dist_class is not None:
                    dist_class = self.dist_class
                else:
                    dist_class = self.model.get_exploration_action_dist_cls()
                try:
                    action_dist = dist_class(
                        action[2]["action_dist_inputs"], self.model
                    )
                except AssertionError:
                    action_dist = dist_class(
                        logits=torch.Tensor(action[2]["action_dist_inputs"])
                    )
                guide_logp = action_dist.logp(torch.Tensor(guide_action))
                action_prob = torch.exp(guide_logp.float())
                if self.config["jsrl"]["deterministic_sample"] == True:
                    if action_prob < 0.5:
                        action_prob = [0.0]
                        guide_logp = [-np.inf]
                    else:
                        action_prob = [1.0]
                        guide_logp = [0.0]
                dist_info = action[2]
                dist_info["action_logp"] = np.array(guide_logp)
                dist_info["action_prob"] = np.array(action_prob)
                action = (act_np, action[1], dist_info)
                return action

        @override(policy)
        def export_checkpoint(self, *args, **kwargs) -> None:
            """Exports Policy checkpoint to a local directory and returns an AIR Checkpoint.

            Args:
                export_dir: Local writable directory to store the AIR Checkpoint
                    information into.
                policy_state: An optional PolicyState to write to disk. Used by
                    `Algorithm.save_checkpoint()` to save on the additional
                    `self.get_state()` calls of its different Policies.
                checkpoint_format: Either one of 'cloudpickle' or 'msgpack'.

            Example:
                >>> from ray.rllib.algorithms.ppo import PPOTorchPolicy
                >>> policy = PPOTorchPolicy(...) # doctest: +SKIP
                >>> policy.export_checkpoint("/tmp/export_dir") # doctest: +SKIP
            """
            # `filename_prefix` should not longer be used as new Policy checkpoints
            # contain more than one file with a fixed filename structure.
            # import pdb;pdb.set_trace()
            guide = self.config["jsrl"]["guide_policy"]
            del self.config["jsrl"]["guide_policy"]
            super().export_checkpoint(*args, **kwargs)
            self.config["jsrl"]["guide_policy"] = guide

    policy = JSRLPolicy
    return policy
