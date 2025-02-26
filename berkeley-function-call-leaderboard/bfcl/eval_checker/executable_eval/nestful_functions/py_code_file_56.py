from typing import Union



def factorial_optimized(num: int) -> Union[int, None, str]:

    """Calculates the factorial of a given integer.

    Args:

        num: The integer to calculate the factorial of.

    Returns:

        The factorial of the input integer if it is non-negative, None if the input is negative,

        or an error message if the input is not an integer.

    """

    if not isinstance(num, int):

        return "Input must be an integer."

    if num < 0:

        return None

    if num in [0, 1]:

        return 1

    result = 1

    for i in range(2, num + 1):

        result *= i

    return result

