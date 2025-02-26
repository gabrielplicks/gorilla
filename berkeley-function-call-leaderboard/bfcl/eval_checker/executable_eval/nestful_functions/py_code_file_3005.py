from collections import defaultdict

from typing import Dict, List



def count_dictionary(lst: List[int]) -> Dict[int, int]:

    """Creates a dictionary whose keys are the integers in the input list and values are the number of times the keys appear in the list.



    Args:

        lst: A list of integers.



    Returns:

        A dictionary with the counts of each integer in the input list.

    """

    counts = defaultdict(int)

    for n in lst:

        counts[n] += 1

    return counts

