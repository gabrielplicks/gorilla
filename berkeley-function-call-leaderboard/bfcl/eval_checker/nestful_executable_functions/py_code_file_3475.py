from typing import List, Tuple



def pythagorean_triples(n: int) -> List[Tuple[int, int, int]]:

    """Generates a list of Pythagorean triples (a, b, c) such that a^2 + b^2 = c^2 and a + b + c <= n.



    Args:

        n: The maximum sum of the entries in the triples.



    Returns:

        A list of Pythagorean triples.

    """

    triples = []

    for a in range(1, n + 1):

        for b in range(1, n + 1):

            c = (a ** 2 + b ** 2) ** 0.5

            if c.is_integer() and a + b + c <= n:

                triples.append((a, b, int(c)))



    return triples

