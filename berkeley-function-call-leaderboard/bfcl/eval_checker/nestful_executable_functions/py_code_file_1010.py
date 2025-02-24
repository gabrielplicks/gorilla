from collections import Counter

from typing import List



def extract_non_unique(lst: List[int]) -> List[int]:

    """Extracts non-unique elements from a list of integers.



    Args:

        lst: A list of integers.



    Returns:

        A list of non-unique elements from the input list.

    """

    counter = Counter(lst)

    return [num for num, count in counter.items() if count > 1]

