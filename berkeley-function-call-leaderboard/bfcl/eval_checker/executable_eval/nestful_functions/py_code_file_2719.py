from typing import Union



def range_sum(a: int, b: int) -> Union[int, float]:

    """Calculates the sum of all numbers in the range from `a` to `b` (inclusive).

    If `b` is less than `a`, the function returns 0.

    Args:

        a: The start of the range.

        b: The end of the range.

    """

    if b < a:

        return 0



    return sum(range(a, b + 1))

