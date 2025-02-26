from typing import Union



def get_sum_of_integers(n: Union[int, float]) -> int:

    """Calculates the sum of all integers from 1 to n (inclusive).



    Args:

        n: A positive integer.



    Returns:

        The sum of all integers from 1 to n (inclusive).

    """

    return (n * (n+1)) // 2

