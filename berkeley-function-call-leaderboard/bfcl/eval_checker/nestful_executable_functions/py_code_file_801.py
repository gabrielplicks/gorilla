from typing import Tuple



def sum_and_difference(a: int, b: int) -> Tuple[int, int]:

    """Computes the sum and difference of two given integers.

    If the sum is even, return the sum and difference, otherwise return the sum and the difference's absolute value.

    Form the result as a tuple of the form (sum, difference).



    Args:

        a: The first integer.

        b: The second integer.



    Returns:

        A tuple containing the sum and difference or the sum and the difference's absolute value.

    """

    sum_ = a + b

    difference = abs(a - b)

    if sum_ % 2 == 0:

        return sum_, difference

    else:

        return sum_, abs(difference)

