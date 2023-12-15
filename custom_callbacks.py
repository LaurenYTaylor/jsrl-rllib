from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
import numpy as np

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
        if 'guide_policy' in policy.config:
            guide_policy_args = policy.config['guide_policy']
            algo, path = guide_policy_args
            trained_algo = algo.from_checkpoint(path)
            policy.config['guide_policy'] = trained_algo

        policy.config['jsrl_threshold'] = 0
        policy.config['jsrl_prev_best'] = -np.inf

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
        print(evaluation_metrics)