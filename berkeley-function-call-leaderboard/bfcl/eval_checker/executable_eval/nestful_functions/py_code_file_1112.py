from typing import Dict, List



def first_n_sum(d: Dict[str, List[float]], x: str, n: int) -> float:

    """Calculates the sum of the first n values of d[x].

    If n is greater than the length of d[x], then returns the sum of all values in d[x].

    If x is not in d, returns 0.

    Args:

        d: A dictionary where d[x] is a list of values.

        x: The key to look up in the dictionary.

        n: The number of values to sum.

    """

    if x in d:

        if n > len(d[x]):

            return sum(d[x])

        else:

            return sum(d[x][:n])

    else:

        return 0

