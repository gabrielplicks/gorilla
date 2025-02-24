from typing import List



def geometric(a: float, r: float, n: int) -> List[float]:

    """

    Returns a list of the first `n` elements of a geometric progression.



    Args:

        a: The first term of the progression.

        r: The common ratio of the progression.

        n: The number of terms to generate.



    Returns:

        A list of the first `n` elements of the progression.

    """

    return [a * r**i for i in range(n)]

