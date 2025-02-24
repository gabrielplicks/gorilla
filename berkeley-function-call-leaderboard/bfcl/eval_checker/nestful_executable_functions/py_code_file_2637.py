from typing import List, Tuple



def expand_ranges(ranges: List[Tuple[int, int]]) -> List[int]:

    """Expands a list of ranges into a sorted list of all the numbers in those ranges.



    Args:

        ranges: A list of tuples representing ranges. Each tuple is of the form `(start, end)`.



    Returns:

        A sorted list of all the numbers in the ranges.

    """

    expanded = []

    for start, end in ranges:

        expanded.extend(range(start, end + 1))

    return sorted(expanded)

