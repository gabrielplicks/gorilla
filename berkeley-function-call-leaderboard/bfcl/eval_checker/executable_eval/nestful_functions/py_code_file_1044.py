from typing import List, Tuple



def min_intervals(intervals: List[Tuple[int, int]]) -> int:

    """Finds the minimum number of intervals required to cover a given set of intervals.



    Args:

        intervals: A list of intervals, where each interval is represented as a tuple (start, end)

            with both values non-negative integers.



    Returns:

        The minimum number of intervals required to cover the entire range.

    """

    intervals = sorted(intervals, key=lambda x: x[0])

    count = 1

    current = intervals[0]

    for start, end in intervals[1:]:

        if current[1] >= start:

            continue

        count += 1

        current = (start, end)

    return count

