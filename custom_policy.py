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
                if ('jsrl_threshold' not in self.config or 
                            self.config['jsrl_threshold'] == np.random.choice([0,1])):
                    print("LEARN ACTION: ", action)
                    return action
                else:
                    guide_action = np.array([self.config['guide_policy'].compute_single_action(input_dict=input_dict,
                            explore=False, timestep=timestep)])
                    seq_lens = torch.ones(len(input_dict), dtype=torch.int32)
                    input_dict.pop('actions')
                    action = self._precomputed_actions_helper(guide_action, input_dict,
                    [], seq_lens, explore, timestep)
                    print("MODIFIED GUIDE ACTION: ", action)
                    return action

        def _precomputed_actions_helper(
            self, precomputed_actions, input_dict, state_batches, seq_lens, explore, timestep
        ):
            """Shared forward pass logic (w/ and w/o trajectory view API).

            Returns:
                A tuple consisting of a) actions, b) state_out, c) extra_fetches.
                The input_dict is modified in-place to include a numpy copy of the computed
                actions under `SampleBatch.ACTIONS`.
            """
            explore = explore if explore is not None else self.config["explore"]
            timestep = timestep if timestep is not None else self.global_timestep

            # Switch to eval mode.
            if self.model:
                self.model.eval()

            extra_fetches = dist_inputs = logp = None

            # New API stack: `self.model` is-a RLModule.
            if isinstance(self.model, RLModule):
                if self.model.is_stateful():
                    # For recurrent models, we need to add a time dimension.
                    if not seq_lens:
                        # In order to calculate the batch size ad hoc, we need a sample
                        # batch.
                        if not isinstance(input_dict, SampleBatch):
                            input_dict = SampleBatch(input_dict)
                        seq_lens = np.array([1] * len(input_dict))
                    input_dict = self.maybe_add_time_dimension(
                        input_dict, seq_lens=seq_lens
                    )
                input_dict = convert_to_torch_tensor(input_dict, device=self.device)

                # Batches going into the RL Module should not have seq_lens.
                if SampleBatch.SEQ_LENS in input_dict:
                    del input_dict[SampleBatch.SEQ_LENS]

                if explore:
                    '''
                    fwd_out = self.model.forward_exploration(input_dict)
                    # For recurrent models, we need to remove the time dimension.
                    fwd_out = self.maybe_remove_time_dimension(fwd_out)
                    '''
                    fwd_out = SampleBatch({SampleBatch.ACTIONS: precomputed_actions})
                    # ACTION_DIST_INPUTS field returned by `forward_exploration()` ->
                    # Create a distribution object.
                    action_dist = None
                    # Maybe the RLModule has already computed actions.
                    if SampleBatch.ACTION_DIST_INPUTS in fwd_out:
                        dist_inputs = fwd_out[SampleBatch.ACTION_DIST_INPUTS]
                        action_dist_class = self.model.get_exploration_action_dist_cls()
                        action_dist = action_dist_class.from_logits(dist_inputs)

                    # If `forward_exploration()` returned actions, use them here as-is.
                    if SampleBatch.ACTIONS in fwd_out:
                        actions = fwd_out[SampleBatch.ACTIONS]
                    # Otherwise, sample actions from the distribution.
                    else:
                        if action_dist is None:
                            raise KeyError(
                                "Your RLModule's `forward_exploration()` method must return"
                                f" a dict with either the {SampleBatch.ACTIONS} key or the "
                                f"{SampleBatch.ACTION_DIST_INPUTS} key in it (or both)!"
                            )
                        actions = action_dist.sample()

                    # Compute action-logp and action-prob from distribution and add to
                    # `extra_fetches`, if possible.
                    if action_dist is not None:
                        logp = action_dist.logp(actions)
                else:
                    #fwd_out = self.model.forward_inference(input_dict)
                    fwd_out = SampleBatch({SampleBatch.ACTIONS: precomputed_actions})
                    # For recurrent models, we need to remove the time dimension.
                    fwd_out = self.maybe_remove_time_dimension(fwd_out)

                    # ACTION_DIST_INPUTS field returned by `forward_exploration()` ->
                    # Create a distribution object.
                    action_dist = None
                    if SampleBatch.ACTION_DIST_INPUTS in fwd_out:
                        dist_inputs = fwd_out[SampleBatch.ACTION_DIST_INPUTS]
                        action_dist_class = self.model.get_inference_action_dist_cls()
                        action_dist = action_dist_class.from_logits(dist_inputs)
                        action_dist = action_dist.to_deterministic()

                    # If `forward_inference()` returned actions, use them here as-is.
                    if SampleBatch.ACTIONS in fwd_out:
                        actions = fwd_out[SampleBatch.ACTIONS]
                    # Otherwise, sample actions from the distribution.
                    else:
                        if action_dist is None:
                            raise KeyError(
                                "Your RLModule's `forward_inference()` method must return"
                                f" a dict with either the {SampleBatch.ACTIONS} key or the "
                                f"{SampleBatch.ACTION_DIST_INPUTS} key in it (or both)!"
                            )
                        actions = action_dist.sample()

                # Anything but actions and state_out is an extra fetch.
                state_out = fwd_out.pop(STATE_OUT, {})
                extra_fetches = fwd_out
            else:
                # Call the exploration before_compute_actions hook.
                
                self.exploration.before_compute_actions(explore=explore, timestep=timestep)
                dist_class = self.dist_class
                import pdb;pdb.set_trace()
                dist_inputs, state_out = self.model(input_dict, state_batches, seq_lens)
                print(dist_inputs)
                if not (
                    isinstance(dist_class, functools.partial)
                    or issubclass(dist_class, TorchDistributionWrapper)
                ):
                    raise ValueError(
                        "`dist_class` ({}) not a TorchDistributionWrapper "
                        "subclass! Make sure your `action_distribution_fn` or "
                        "`make_model_and_action_dist` return a correct "
                        "distribution class.".format(dist_class.__name__)
                    )
                action_dist = dist_class(dist_inputs, self.model)

                # Get the exploration action from the forward results.
                actions = torch.Tensor(precomputed_actions)
                logp = action_dist.logp(actions)
                input_dict['actions'] = actions

            # Add default and custom fetches.
            if extra_fetches is None:
                extra_fetches = self.extra_action_out(
                    input_dict, state_batches, self.model, action_dist
                )
            

            # Action-dist inputs.
            if dist_inputs is not None:
                extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs

            # Action-logp and action-prob.
            if logp is not None:
                extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp.float())
                extra_fetches[SampleBatch.ACTION_LOGP] = logp

            # Update our global timestep by the batch size.
            self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])
            return convert_to_numpy((actions, state_out, extra_fetches))

                    

        '''
        @override(policy.__parent__)
        def action_sampler_fn(
            self,
            model: ModelV2,
            *,
            obs_batch: TensorType,
            state_batches: TensorType,
            **kwargs,
        ) -> Tuple[TensorType, TensorType, TensorType, List[TensorType]]:
            """Custom function for sampling new actions given policy.

            Args:
                model: Underlying model.
                obs_batch: Observation tensor batch.
                state_batches: Action sampling state batch.

            Returns:
                Sampled action
                Log-likelihood
                Action distribution inputs
                Updated state
            """
            # WHY IS THIS NOT BEING CALLED
            if 'guide_policy' in self.config:
                print(self.config['guide_policy'])
            return None, None, None, None
        '''

    policy = JSRLPolicy
    return policy

