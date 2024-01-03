from custom_policy import make_custom_policy
from ray.rllib.utils.annotations import override


def make_custom_algorithm(algorithm):
    """Returns a custom algorithm that subclasses the chosen algorithm, and uses the custom policy.

    Args:
        algorithm (ray.rllib.algorithm): The chosen algorithm.

    Returns:
        CustomAlgo: The algorithm that uses the custom policy.
    """
    policy = make_custom_policy(algorithm)
    policy.__name__ = "JSRLPolicy"

    class CustomAlgo(algorithm):
        @override(algorithm)
        def get_default_policy_class(self, config):
            return policy

    algo = CustomAlgo

    return algo
