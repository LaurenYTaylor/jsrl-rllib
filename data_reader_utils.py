from ray.rllib.offline import ShuffledInput, InputReader, IOContext
from custom_datareader import CustomD4RLReader, CustomJsonReader


def json_input_creator(ioctx: IOContext) -> InputReader:
    """
    The input creator method can be used in the input registry or set as the
    config["input"] parameter.

    Args:
        ioctx: use this to access the `input_config` arguments.

    Returns:
        instance of ShuffledInput to work with some offline rl algorithms
    """
    return ShuffledInput(CustomJsonReader(ioctx))


def d4rl_input_creator(ioctx: IOContext) -> InputReader:
    """
    The input creator method can be used in the input registry or set as the
    config["input"] parameter.

    Args:
        ioctx: use this to access the `input_config` arguments.

    Returns:
        instance of ShuffledInput to work with some offline rl algorithms
    """

    inputs = ioctx.input_config["env"]
    return ShuffledInput(CustomD4RLReader(inputs))
