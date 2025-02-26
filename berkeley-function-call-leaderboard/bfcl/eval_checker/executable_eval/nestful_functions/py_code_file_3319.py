from typing import List, Tuple



def extract_slice(a: List[int], d: dict) -> List[Tuple[int, int]]:

    """Extracts a slice of a list between the second value and the fifth value in a dictionary.

    Args:

        a: The list to extract the slice from.

        d: The dictionary containing the indices to extract the slice from.

    """

    start = d['b']

    end = d['e']

    return [(i, a[i]) for i in range(start, end)]

