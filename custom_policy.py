import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import tree  # pip install dm_tree

from ray.rllib.utils.annotations import (
    override,
)
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import (
    TensorType,
)

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
            action = super().compute_actions_from_input_dict(
                input_dict, explore, timestep, **kwargs
            )
            if (
                "current_horizon" not in self.config["jsrl"]
                or self.config["jsrl"]["horizon_fn"](kwargs["episodes"][-1], input_dict)
                > self.config["jsrl"]["current_horizon"]
            ):
                # During evaluation, keep track of agent_type
                if "agent_type" in self.config["jsrl"] and not (
                    len(self.config["jsrl"]["agent_type"]) == 1
                    and self.config["jsrl"]["agent_type"][0] is None
                ):
                    self.config["jsrl"]["agent_type"].append(1)
                return action
            else:
                if "agent_type" in self.config["jsrl"] and not (
                    len(self.config["jsrl"]["agent_type"]) == 1
                    and self.config["jsrl"]["agent_type"][0] is None
                ):
                    self.config["jsrl"]["agent_type"].append(0)
                if "actions" in input_dict:
                    input_dict.pop("actions")

                # The following makes a guide action computed from the pre-trained policy have the same information as a learning action
                guide_action = np.array(
                    [
                        self.config["jsrl"]["guide_policy"].compute_single_action(
                            input_dict=input_dict, explore=False, timestep=timestep
                        )
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

    policy = JSRLPolicy
    return policy
