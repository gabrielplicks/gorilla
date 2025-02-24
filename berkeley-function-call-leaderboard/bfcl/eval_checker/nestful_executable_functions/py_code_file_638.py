from typing import Union



def are_close(x: Union[float, int], y: Union[float, int], epsilon: float = 1e-6) -> bool:

    """Checks if two floating-point values `x` and `y` are close enough to be considered equal.



    Args:

        x: The first floating-point value.

        y: The second floating-point value.

        epsilon: The threshold value.



    Returns:

        True if their absolute difference is less than or equal to `epsilon`, and False otherwise.

    """

    return abs(x - y) <= epsilon

