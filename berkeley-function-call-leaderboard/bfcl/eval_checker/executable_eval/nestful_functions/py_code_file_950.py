from typing import List, Tuple



def sort_and_count(strings: List[str]) -> List[Tuple[str, int]]:

    """Sorts a list of strings alphabetically while keeping track of the number of times each string appears.



    Args:

        strings: A list of strings to be sorted and counted.



    Returns:

        A list of tuples where each tuple contains the original string and its count.

        If two strings have the same count, they are sorted lexicographically to maintain their original order.

    """

    counts: dict[str, int] = {}

    for string in strings:

        counts[string] = counts.get(string, 0) + 1



    sorted_keys = sorted(counts, key=lambda x: (-counts[x], x))



    return [(key, counts[key]) for key in sorted_keys]

