from typing import Union



def sum_total(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:

    """Calculates the sum of two numbers.



    Args:

        a: The first number.

        b: The second number.



    Returns:

        The sum of `a` and `b`.

    """

    return a + b



def sum_total_with_c(a: Union[int, float], b: Union[int, float], c: Union[int, float]) -> Union[int, float]:

    """Calculates the sum of three numbers.



    Args:

        a: The first number.

        b: The second number.

        c: The third number.



    Returns:

        The sum of `a`, `b`, and `c`.

    """

    return sum_total(a, b) + c

