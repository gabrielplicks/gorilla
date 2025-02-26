from typing import Union



def within_tolerance(actual: Union[int, float], expected: Union[int, float], tolerance: float = 0.001) -> bool:

    """Checks if the absolute difference between actual and expected is less than or equal to tolerance * expected.



    Args:

        actual: The actual value.

        expected: The expected value.

        tolerance: The ratio of tolerance (default is 0.001).

    """

    return abs(actual - expected) <= tolerance * expected

