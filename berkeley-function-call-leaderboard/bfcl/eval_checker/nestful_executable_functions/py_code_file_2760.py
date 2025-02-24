from typing import Optional



def calculate_power_of_two(num: int) -> Optional[int]:

    """

    Calculates the power of two for a given integer. If the result exceeds the maximum

    value of an integer, returns `None`.

    Args:

        num: The integer to calculate the power of two for.

    """

    if num == 0:

        return 0

    elif num < 0:

        return None

    elif num == 1:

        return num

    else:

        result = 1

        while result <= num:

            result <<= 1

        return result >> 1

