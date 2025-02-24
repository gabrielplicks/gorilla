from typing import List, Tuple



def match_intervals(nums: List[int], intervals: List[Tuple[int, int]]) -> List[int]:

    """

    Returns a list of integers that match at least one of the given intervals.



    Args:

        nums: A list of integers.

        intervals: A list of tuples representing integer intervals.

    """

    return [num for num in nums if any(interval[0] <= num <= interval[1] for interval in intervals)]

