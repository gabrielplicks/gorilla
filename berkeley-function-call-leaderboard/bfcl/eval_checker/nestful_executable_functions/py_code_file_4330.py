from typing import Tuple



def power_modulo(n: int, p: int) -> int:

    """Calculates the result of `n` to the power of `p` modulo 10^9 + 7.



    Args:

        n: A positive integer.

        p: A positive integer.



    Returns:

        The result of `n` to the power of `p` modulo 10^9 + 7.

    """

    assert isinstance(n, int) and n > 0, "n must be a positive integer"

    assert isinstance(p, int) and p > 0, "p must be a positive integer"



    result = 1

    for _ in range(p):

        result = (result * n) % (10 ** 9 + 7)

    return result

