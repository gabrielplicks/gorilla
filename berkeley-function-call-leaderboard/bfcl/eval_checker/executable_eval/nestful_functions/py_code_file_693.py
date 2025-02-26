from typing import Union



def is_near_abs(a: Union[int, float], b: Union[int, float], threshold: Union[int, float]) -> bool:

    """Checks if the absolute difference between a and b is less than or equal to threshold.



    Args:

        a: The first number.

        b: The second number.

        threshold: The maximum allowed difference.



    Returns:

        True if the absolute difference is less than or equal to threshold, False otherwise.

    """

    diff = abs(a - b)

    return diff <= threshold

