from typing import Union



def triangular_number(n: Union[int, float]) -> int:

    """Calculates the nth triangular number.



    A triangular number is the sum of all integers from 1 to `n`, inclusive.



    Args:

        n: A positive number.

    """

    result = 0



    for i in range(1, int(n) + 1):

        result += i



    return result

