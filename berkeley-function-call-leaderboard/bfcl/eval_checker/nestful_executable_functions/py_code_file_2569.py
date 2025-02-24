from typing import List



def generate_geometric_progression(a: float, r: float, n: int) -> List[float]:

    """Generates terms of a geometric progression given a starting value `a`, a common ratio `r`, and a number of terms `n`.



    Args:

        a: The starting value of the geometric progression.

        r: The common ratio of the geometric progression.

        n: The number of terms to generate.



    Returns:

        A list of `n` terms, where each term is the product of the previous term and the common ratio `r`.

    """

    return [a * r**i for i in range(n)]

