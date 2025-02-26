from typing import Union



def find_sum_of_squares(a: Union[int, float], b: Union[int, float]) -> int:

    """Calculates the sum of the squares of two numbers.



    Args:

        a: The first number.

        b: The second number.



    Raises:

        AssertionError: If the inputs are not valid integers or not in the range 0 < a, b < 100.

    """

    assert isinstance(a, int) and isinstance(b, int), "Inputs must be integers"

    assert 0 < a < 100 and 0 < b < 100, "Inputs must be in range 0 < a, b < 100"

    return a ** 2 + b ** 2

