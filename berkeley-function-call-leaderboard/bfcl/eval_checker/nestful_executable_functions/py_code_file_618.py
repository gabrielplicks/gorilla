from typing import Union



def relative_error(s: Union[int, float], us: Union[int, float]) -> float:

    """Calculates the relative error of a measurement of size s, given the uncertainty us in the measurement.

    Args:

        s: The size of the measurement.

        us: The uncertainty in the measurement.

    """

    assert s > 0, "Size must be positive"

    return us / s

