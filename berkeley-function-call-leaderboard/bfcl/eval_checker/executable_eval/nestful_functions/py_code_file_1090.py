from typing import List



def sort_and_merge_intervals(intervals: List[List[int]]) -> List[List[int]]:

    """Sorts a list of intervals and merges overlapping intervals.



    Args:

        intervals: A list of intervals, where each interval is a list of two integers.



    Returns:

        A list of merged intervals.

    """

    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged_intervals = []



    for interval in sorted_intervals:

        if not merged_intervals or interval[0] > merged_intervals[-1][1]:

            merged_intervals.append(interval)

        else:

            merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])



    return merged_intervals

