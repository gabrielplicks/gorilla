from collections import Counter

from typing import Dict, List



def count_distinct_integers(nums: List[int]) -> Dict[int, int]:

    """Counts the occurrences of each distinct integer in a list.



    Args:

        nums: A list of integers.



    Returns:

        A dictionary where the keys are the distinct integers in the list,

        and the values are the number of occurrences of each key.

    """

    return {num: count for num, count in Counter(nums).items()}

