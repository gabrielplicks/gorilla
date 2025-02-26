from typing import List

import itertools



def get_all_str_combinations(s: str, k: int) -> List[str]:

    """

    Returns a list of all combinations of unique substrings of size k from s sorted in ascending order.

    If k is greater than the length of s, return an empty list.



    Args:

        s: The input string.

        k: The size of the substrings.



    Returns:

        A list of all combinations of unique substrings of size k from s sorted in ascending order.

    """

    if k > len(s):

        return []

    return sorted(set(''.join(c) for c in itertools.combinations(s, k)))

