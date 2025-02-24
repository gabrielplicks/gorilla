from typing import List, Tuple

from collections import Counter



def sort_unique_by_frequency(array: List[int]) -> List[Tuple[int, int]]:

    """Sorts the unique integers in an array in descending order of their occurrence frequency.



    Args:

        array: A list of integers.



    Returns:

        A list of tuples, where each tuple contains an integer and its frequency.

        The integers are sorted in descending order of their frequency, with the integers

        that appear more often listed first.

    """

    counter = Counter(array)

    return sorted(counter.items(), key=lambda x: (x[1], -x[0]), reverse=True)

