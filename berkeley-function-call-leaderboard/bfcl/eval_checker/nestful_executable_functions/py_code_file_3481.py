from typing import List, Tuple



def find_interval_distance(intervals: List[Tuple[float, float]]) -> float:

    """Calculates the total distance between adjacent intervals in a list of intervals.

    Args:

        intervals: A list of intervals, where each interval is represented as a tuple (start, end).

    """

    total_distance = 0

    for interval in intervals:

        start, end = interval

        total_distance += end - start

    return total_distance

