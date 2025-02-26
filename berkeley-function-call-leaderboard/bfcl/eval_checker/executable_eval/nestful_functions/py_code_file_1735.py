from typing import List



def divide_tasks(tasks: List[int], n: int) -> List[List[int]]:

    """Divides a list of tasks into equal-sized sublists of size `n`.

    Args:

        tasks: The list of tasks to divide.

        n: The size of each sublist.

    Returns:

        A nested list of sublists and the number of sublists.

    """

    sublists = []

    sublists_count = 0

    for i in range(0, len(tasks), n):

        sublist = tasks[i:i+n]

        sublists.append(sublist)

        sublists_count += 1

    return sublists, sublists_count

