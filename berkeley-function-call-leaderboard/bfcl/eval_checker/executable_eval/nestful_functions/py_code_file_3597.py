from typing import Tuple



def remap_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:

    """Maps a value from one range to another.



    Args:

        x: The input value to be remapped.

        in_min: The minimum value of the input range.

        in_max: The maximum value of the input range.

        out_min: The minimum value of the output range.

        out_max: The maximum value of the output range.

    """

    return out_min + ((x - in_min) / (in_max - in_min)) * (out_max - out_min)

