from typing import List, Tuple

import itertools



def group_and_sort_by_second(tuples: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:

    """Groups a list of tuples by the second element and sorts the resulting groups based on the first element.



    Args:

        tuples: A list of tuples to be grouped and sorted.



    Returns:

        A list of groups, where each group is a list of tuples.

    """

    sorted_tuples = sorted(tuples, key=lambda x: x[1])

    groups = [list(group) for _, group in itertools.groupby(sorted_tuples, key=lambda x: x[1])]

    for group in groups:

        group.sort(key=lambda x: x[0])



    return groups

