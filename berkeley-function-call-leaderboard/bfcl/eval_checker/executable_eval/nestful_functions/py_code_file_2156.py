from typing import Dict, List



def count_integers(lst: List[int]) -> Dict[int, int]:

    """

    Counts the number of times each integer appears in a list.



    Args:

        lst: A list of integers.



    Returns:

        A dictionary whose keys are the integers and values are the number of times the integers appear in the list.

    """

    counts = {}

    for num in lst:

        if num in counts:

            counts[num] += 1

        else:

            counts[num] = 1

    return counts

