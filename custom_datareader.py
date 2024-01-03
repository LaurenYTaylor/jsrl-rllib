from ray.rllib.offline import D4RLReader, IOContext, JsonReader
from ray.rllib.offline.d4rl_reader import _convert_to_batch
from ray.rllib.utils.annotations import override
from copy import deepcopy

from ray.rllib.policy.sample_batch import (
    SampleBatch,
    concat_samples,
    convert_ma_batch_to_sample_batch,
)
from ray.rllib.utils.annotations import override

from ray.rllib.utils.typing import SampleBatchType
import d4rl_env_maker


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
    def __init__(self, inputs: str, ioctx: IOContext = None):
        """Initializes a CustomD4RLReader instance.

        Args:
            inputs: String corresponding to the D4RL environment creation function.
            ioctx: Current IO context object.
        """
        import d4rl

        env_name = inputs.split(".")[1]
        env_fn = getattr(d4rl_env_maker, env_name)
        self.env = env_fn().env
        self.dataset = _convert_to_batch(d4rl.qlearning_dataset(self.env))
        assert self.dataset.count >= 1
        self.counter = 0
