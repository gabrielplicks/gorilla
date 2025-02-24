from typing import List, Tuple



def transform_tuples(lst: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:

    """Transforms a list of (x, y) tuples into a list of (x, y, x + y) tuples.



    Args:

        lst: A list of (x, y) tuples.



    Returns:

        A list of (x, y, x + y) tuples.

    """

    return [(x, y, x + y) for x, y in lst]

