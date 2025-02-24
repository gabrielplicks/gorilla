from typing import Union



def log2_floor(num: Union[int, float]) -> int:

    """Calculates the floor of the log base 2 of a given positive integer.



    Args:

        num: The input number.



    Returns:

        The floor of the log base 2 of the input number.

    """

    result = 0

    while num >= 1:

        num >>= 1  # Shift the input number one position to the right (divide by 2)

        result += 1

    return result - 1  # Subtract 1 to compensate for the extra increment at the end of the loop

