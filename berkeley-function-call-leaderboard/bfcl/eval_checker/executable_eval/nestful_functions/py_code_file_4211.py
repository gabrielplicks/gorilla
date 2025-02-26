from typing import Union



def sum_consecutive_integers(n: Union[int, float]) -> Union[int, float]:

    """Calculates the sum of consecutive integers from 1 to a given number, inclusive.

    Args:

        n: The given number.

    """

    sum = n * (n + 1) // 2

    print(f'The sum of consecutive integers from 1 to {n} is {sum}.')

    return sum

