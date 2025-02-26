from typing import Union



def angle_diff(a: Union[int, float], b: Union[int, float]) -> float:

    """Calculates the absolute difference between two angles, a and b, but ensures that the result is within the range of [-180, 180).



    The function handles negative angles correctly and returns a value in degrees.



    Args:

        a: The first angle.

        b: The second angle.

    """

    diff = abs(a - b)

    return diff if diff <= 180 else 360 - diff

