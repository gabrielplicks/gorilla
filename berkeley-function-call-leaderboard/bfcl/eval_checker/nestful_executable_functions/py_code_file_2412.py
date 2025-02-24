from typing import List, Tuple



def f_list(tuples: List[Tuple[int, int]]) -> List[int]:

    """Calculates the value of f(a, b) for each tuple in the list.



    Args:

        tuples: A list of tuples (a, b).



    Returns:

        A list of values f(a, b) for each tuple.

    """

    results = []

    for a, b in tuples:

        if a > b:

            result = max(a, b)

        elif a < b:

            result = min(a, b)

        else:

            result = a + b

        results.append(result)

    return results

