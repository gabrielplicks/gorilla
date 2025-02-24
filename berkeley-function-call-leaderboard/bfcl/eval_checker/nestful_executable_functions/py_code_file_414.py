from typing import List, Tuple



def intersects(intervals: List[Tuple[int, int]]) -> bool:

    """Checks if any two intervals in the given list intersect.



    Args:

        intervals: A list of time intervals, where each interval is represented by a tuple of two integers.



    Returns:

        True if any two intervals intersect, False otherwise.

    """

    for i in range(len(intervals) - 1):

        for j in range(i + 1, len(intervals)):

            a, b = intervals[i]

            c, d = intervals[j]

            if a <= d and c <= b:

                return True

    return False

