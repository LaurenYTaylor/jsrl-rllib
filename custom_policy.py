import copy
import functools
import logging
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree

import ray
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module import RLModule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.utils import NullContextManager, force_list
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.error import ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
    NUM_AGENT_STEPS_TRAINED,
    NUM_GRAD_UPDATES_LIFETIME,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    GradInfoDict,
    ModelGradients,
    ModelWeights,
    PolicyState,
    TensorStructType,
    TensorType,
)

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)

def make_custom_policy(algo):
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
                action = super().compute_actions_from_input_dict(input_dict, explore, timestep, **kwargs)
                if ("current_horizon" not in self.config["jsrl"] or 
                            self.config["jsrl"]["horizon_fn"](kwargs["episodes"][-1], input_dict) 
                            > self.config["jsrl"]["current_horizon"]):
                    if ("agent_type" in self.config["jsrl"] and not (
                        len(self.config["jsrl"]["agent_type"])==1 and 
                        self.config["jsrl"]["agent_type"][0] is None)):
                        self.config["jsrl"]["agent_type"].append(1)
                    return action
                else:
                    if ("agent_type" in self.config["jsrl"] and not (
                        len(self.config["jsrl"]["agent_type"])==1 and 
                        self.config["jsrl"]["agent_type"][0] is None)):
                        self.config["jsrl"]["agent_type"].append(0)
                    if 'actions' in input_dict:
                        input_dict.pop('actions')
                    guide_action = np.array([self.config['jsrl']['guide_policy'].compute_single_action(input_dict=input_dict,
                            explore=False, timestep=timestep)])
                    act_np = np.array(guide_action)
                    if self.dist_class is not None:
                        dist_class = self.dist_class
                    else:
                        dist_class = self.model.get_exploration_action_dist_cls()
                    try:
                        action_dist = dist_class(action[2]['action_dist_inputs'], self.model)
                    except AssertionError:
                        action_dist = dist_class(logits=torch.Tensor(action[2]['action_dist_inputs']))
                    guide_logp = action_dist.logp(torch.Tensor(guide_action))
                    action_prob = torch.exp(guide_logp.float())
                    if self.config['jsrl']['deterministic_sample'] == True:
                        if action_prob < 0.5:
                            action_prob = [0.]
                            guide_logp = [-np.inf]
                        else:
                            action_prob = [1.]
                            guide_logp = [0.]
                    dist_info = action[2]
                    dist_info['action_logp'] = np.array(guide_logp)
                    dist_info['action_prob'] = np.array(action_prob)
                    action = (act_np, action[1], dist_info)
                    return action
                
    policy = JSRLPolicy
    return policy

