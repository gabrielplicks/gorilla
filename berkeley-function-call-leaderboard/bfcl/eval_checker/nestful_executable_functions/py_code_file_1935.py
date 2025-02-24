from typing import List, Tuple



def count_runs(numbers: List[int]) -> List[Tuple[int, int, int]]:

    """

    Returns a list of tuples (start, end, count) where each tuple represents a contiguous run of the same number in the input list.



    Args:

        numbers: A list of numbers.



    Returns:

        A list of tuples (start, end, count) where each tuple represents a contiguous run of the same number in the input list.

    """

    runs = []

    for i, num in enumerate(numbers):

        if len(runs) == 0 or num != runs[-1][2]:

            runs.append((i, i, num))

        else:

            runs[-1] = (runs[-1][0], i, num)



    return runs

