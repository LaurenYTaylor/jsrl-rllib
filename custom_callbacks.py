from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
import numpy as np
from collections import deque


class UpdateThresholdCallback(DefaultCallbacks):
    """Callback that updates the threshold that determines whether to use the guide or learning policy.
    This is updated depending on evaluation results - the algorithm progresses to using more learning policy
    if the evaluation results are good.

    Args:
        DefaultCallbacks (ray.rllib.algorithms.callbacks): Default callbacks.
    """

    def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
        """Callback run whenever a new policy is added to an algorithm.
        Initialises the JSRL variables horizon_idx, current_horizon, thresholds and prev_best.

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
            trained_algo = Policy.from_checkpoint(path + "/policies/default_policy")
            policy.config["jsrl"]["guide_policy"] = trained_algo

            if "jsrl_prev_best" not in policy.config["jsrl"]:
                policy.config["jsrl"]["jsrl_prev_best"] = -np.inf
            policy.config["jsrl"]["thresholds"] = np.linspace(
                policy.config["jsrl"]["init_horizon"],
                0,
                policy.config["jsrl"]["curriculum_stages"],
            )
            print(policy.config["jsrl"]["thresholds"])
            policy.config["jsrl"]["threshold_idx"] = 0
            policy.config["jsrl"]["current_horizon"] = policy.config["jsrl"][
                "thresholds"
            ][policy.config["jsrl"]["threshold_idx"]]
            policy.config["jsrl"]["rolling_mean_rews"] = deque(
                maxlen=policy.config["jsrl"]["rolling_mean_n"]
            )

    def on_evaluate_start(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Callback before evaluation starts.

        This method gets called at the beginning of Algorithm.evaluate().
        Initialises an empty agent_type list for tracking how much the guide vs. learning agent is used.

        Args:
            algorithm: Reference to the algorithm instance.
            kwargs: Forward compatibility placeholder.
        """

        def clr_agent_type(p, p_id):
            p.config["jsrl"]["agent_type"] = []
            p.config["jsrl"]["mean_threshold"] = []

        algorithm.evaluation_workers.foreach_policy(clr_agent_type)

    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        """Runs when the evaluation is done.

        Runs at the end of Algorithm.evaluate(). Determines if the JSRL horizon should be progressed,
        which occurs if the evaluation rolling mean surpasses the previous best score.

        Args:
            algorithm: Reference to the algorithm instance.
            evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        base_policy = algorithm.get_policy()

        if (
            base_policy.config["jsrl"]["threshold_idx"]
            == len(base_policy.config["jsrl"]["thresholds"]) - 1
        ):
            # if algorithm is already on the final curriculum stage, continue
            return

        mean_reward = evaluation_metrics["evaluation"]["sampler_results"][
            "episode_reward_mean"
        ]
        base_policy.config["jsrl"]["rolling_mean_rews"].append(mean_reward)

        rolling_mean = np.mean(base_policy.config["jsrl"]["rolling_mean_rews"])
        if not np.isinf(base_policy.config["jsrl"]["jsrl_prev_best"]):
            prev_best = (
                base_policy.config["jsrl"]["jsrl_prev_best"]
                - base_policy.config["jsrl"]["tolerance"]
                * base_policy.config["jsrl"]["jsrl_prev_best"]
            )
        else:
            prev_best = base_policy.config["jsrl"]["jsrl_prev_best"]

        def update_jsrl_stats(policy, _):
            policy.config["jsrl"]["threshold_idx"] += 1
            policy.config["jsrl"]["current_horizon"] = policy.config["jsrl"][
                "thresholds"
            ][policy.config["jsrl"]["threshold_idx"]]
            policy.config["jsrl"]["jsrl_prev_best"] = rolling_mean

        # Update the jsrl config for each policy (if multiple workers),
        # for both evaluation and rollout workers
        if (
            len(base_policy.config["jsrl"]["rolling_mean_rews"])
            == base_policy.config["jsrl"]["rolling_mean_n"]
            and rolling_mean > prev_best
        ):
            algorithm.workers.foreach_policy(update_jsrl_stats)
            algorithm.evaluation_workers.foreach_policy(update_jsrl_stats)

        if not np.isinf(base_policy.config["jsrl"]["jsrl_prev_best"]):
            evaluation_metrics["jsrl/current_best"] = base_policy.config["jsrl"][
                "jsrl_prev_best"
            ]
        evaluation_metrics["jsrl/current_horizon_idx"] = base_policy.config["jsrl"][
            "threshold_idx"
        ]
        evaluation_metrics["jsrl/current_horizon"] = base_policy.config["jsrl"][
            "current_horizon"
        ]

        def get_eval_stats(policy, _):
            agent_type = policy.config["jsrl"]["agent_type"]
            mean_thresholds = policy.config["jsrl"]["mean_threshold"]
            policy.config["jsrl"]["agent_type"] = []
            policy.config["jsrl"]["mean_threshold"] = []
            return agent_type, mean_thresholds

        # Clear the agent_type list for the eval workers
        agent_type, mean_threshold = algorithm.evaluation_workers.foreach_policy(
            get_eval_stats
        )[-1]
        evaluation_metrics["jsrl/mean_agent_type"] = np.mean(agent_type)
        evaluation_metrics["jsrl/mean_threshold"] = np.mean(mean_threshold)
