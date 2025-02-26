import math



def compare_two_values(x: float, y: float) -> int:

    """Compares two values without using any comparison operator.



    Args:

        x: The first value to compare.

        y: The second value to compare.



    Returns:

        1 if x > y, -1 if x < y, and 0 otherwise.

    """

    diff = x - y

    return math.copysign(1.0, diff)

