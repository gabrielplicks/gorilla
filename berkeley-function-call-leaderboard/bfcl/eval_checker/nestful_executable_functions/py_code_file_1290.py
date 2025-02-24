import numpy as np



def clamp_v2(value: float, min_value: float, max_value: float) -> float:

    """

    Clamps a value between a minimum and maximum value.



    Args:

        value: The value to clamp.

        min_value: The minimum value.

        max_value: The maximum value.

    """

    return np.maximum(np.minimum(value, max_value), min_value)



def clamp_v3(value: float, minval: float, maxval: float) -> float:

    """

    Clamps a value between a minimum and maximum value.



    Args:

        value: The value to clamp.

        minval: The minimum value.

        maxval: The maximum value.

    """

    return clamp_v2(value, minval, maxval)

