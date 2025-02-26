from typing import Union



def find_perfect_squares_num(n: Union[int, float]) -> int:

    """Finds the number of perfect squares less than or equal to `n`.



    Args:

        n: A positive integer or float.



    Returns:

        The number of perfect squares less than or equal to `n`.

    """

    return sum(1 for i in range(1, int(n ** 0.5) + 1) if i ** 2 <= n)

