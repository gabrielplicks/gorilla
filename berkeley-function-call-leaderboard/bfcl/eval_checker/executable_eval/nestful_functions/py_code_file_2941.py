from typing import Union



def exp_v2(x: Union[int, float], n: int) -> Union[int, float]:

    """Computes x^n in fewer than n multiplications using the square-and-multiply approach.



    Args:

        x: The base of the exponentiation.

        n: The exponent.



    Returns:

        The result of x^n.

    """

    if n <= 1:

        return x

    elif n % 2 == 0:

        return exp_v2(x, n // 2) * exp_v2(x, n // 2)

    else:

        return x * exp_v2(x, n - 1)

