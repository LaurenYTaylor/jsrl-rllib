from custom_policy import make_custom_policy
from ray.rllib.utils.annotations import override

import ray
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
import ray.cloudpickle as pickle

from ray.rllib.utils.annotations import (
    override,
)
from ray.rllib.utils.checkpoints import (
    CHECKPOINT_VERSION,
    CHECKPOINT_VERSION_LEARNER,
    get_checkpoint_info,
    try_import_msgpack,
)
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    Deprecated,
    deprecation_warning,
)

from ray.rllib.utils.policy import validate_policy_id

from ray.rllib.utils.typing import (
    PolicyState,
)


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
