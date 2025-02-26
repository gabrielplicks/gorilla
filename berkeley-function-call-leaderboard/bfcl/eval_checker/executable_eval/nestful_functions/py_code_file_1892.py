from typing import Dict, List, Tuple



def sort_by_count(lst: List[int]) -> List[Tuple[int, int]]:

    """Returns a list of tuples with each tuple containing a number and its count,

    where the list is sorted in ascending order by count, and if the counts are the same,

    then sort by numbers.



    Args:

        lst: The input list of numbers.



    Returns:

        A list of tuples with each tuple containing a number and its count.

    """

    counts: Dict[int, int] = {}

    for item in lst:

        counts[item] = counts.get(item, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda x: (x[1], x[0]))

    return [(k, v) for k, v in sorted_counts]

