from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
import numpy as np
from collections import deque

class UpdateThresholdCallback(DefaultCallbacks):
    def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
        """Callback run whenever a new policy is added to an algorithm.

        Args:
            policy_id: ID of the newly created policy.
            policy: The policy just created.
        """

        # Load the guide_policy here because you can't
        # put an algorithm object into the initial config
        # (pickling error)
        if "jsrl" in policy.config:
            guide_policy_args = policy.config["jsrl"]["guide_policy"]
            algo, path = guide_policy_args
            trained_algo = algo.from_checkpoint(path)
            policy.config["jsrl"]["guide_policy"] = trained_algo

            
            if 'jsrl_prev_best' not in policy.config['jsrl']:
                policy.config['jsrl']['jsrl_prev_best'] = -np.inf
            policy.config["jsrl"]['thresholds'] = np.linspace(policy.config['jsrl']['max_horizon'],
                                        0,
                                        policy.config['jsrl']['curriculum_stages'])
            policy.config["jsrl"]['threshold_idx'] = 0
            policy.config['jsrl']['current_horizon'] = policy.config["jsrl"]['thresholds'][policy.config["jsrl"]['threshold_idx']]
            policy.config["jsrl"]["rolling_mean_rews"] = deque(maxlen=policy.config["jsrl"]["rolling_mean_n"])

    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        """Runs when the evaluation is done.

        Runs at the end of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        if policy.config["jsrl"]["threshold_idx"] == len(policy.config["jsrl"]['thresholds']) - 1:
            # if algorithm is already on the final curriculum stage, continue
            return
        
        policy = algorithm.get_policy()
        mean_reward = evaluation_metrics['evaluation']['sampler_results']['episode_reward_mean']
        policy.config["jsrl"]["rolling_mean_rews"].append(mean_reward)
        
        rolling_mean = np.mean(policy.config["jsrl"]["rolling_mean_rews"])
        if not np.isinf(policy.config["jsrl"]["jsrl_prev_best"]):
            prev_best = policy.config["jsrl"]["jsrl_prev_best"]-policy.config["jsrl"]["tolerance"]*policy.config["jsrl"]["jsrl_prev_best"]
        else:
            prev_best = policy.config["jsrl"]["jsrl_prev_best"]
            
        if (len(policy.config["jsrl"]["rolling_mean_rews"]) == policy.config["jsrl"]["rolling_mean_n"] and 
            rolling_mean > prev_best):
            policy.config["jsrl"]["threshold_idx"] += 1
            policy.config['jsrl']['current_horizon'] = policy.config["jsrl"]['thresholds'][policy.config["jsrl"]['threshold_idx']]
            policy.config["jsrl"]["jsrl_prev_best"] = rolling_mean

        evaluation_metrics["jsrl/current_best"] = policy.config["jsrl"]["jsrl_prev_best"]
        evaluation_metrics["jsrl/current_horizon_idx"] = policy.config["jsrl"]["threshold_idx"]
        evaluation_metrics["jsrl/current_horizon"] = policy.config["jsrl"]["current_horizon"]