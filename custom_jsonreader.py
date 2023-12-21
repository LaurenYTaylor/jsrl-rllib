from copy import deepcopy
from ray.rllib.offline import JsonReader, IOContext

from ray.rllib.policy.sample_batch import (
    SampleBatch,
    concat_samples,
    convert_ma_batch_to_sample_batch,
)
from ray.rllib.utils.annotations import override

from ray.rllib.utils.typing import SampleBatchType


class CustomJsonReader(JsonReader):  
    def __init__(self, ioctx: IOContext):
        """
        The constructor must take an IOContext to be used in the input config.
        Args:
            ioctx: use this to access the `input_config` arguments.
        """
        super().__init__(ioctx.input_config["input_files"], ioctx)
    
    @override(JsonReader)
    def _postprocess_if_needed(self, batch: SampleBatchType) -> SampleBatchType:
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