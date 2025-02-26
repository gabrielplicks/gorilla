from typing import Union



def in_to_pt(inches: Union[int, float]) -> float:

    """Converts inches to points.

    Args:

        inches: The value in inches to be converted to points.

    """

    return inches * 72

