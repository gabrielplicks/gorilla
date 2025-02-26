from datetime import datetime

from typing import List



def get_intervals(datetimes: List[datetime]) -> List[tuple]:

    """Calculates the intervals between consecutive datetimes in a list.



    Args:

        datetimes: A list of datetimes.



    Returns:

        A list of tuples, where each tuple contains the difference in days and the actual datetime interval between two consecutive datetimes.

    """

    intervals = []

    for i in range(1, len(datetimes)):

        interval = datetimes[i] - datetimes[i-1]

        interval_tuple = (interval.days, interval)

        intervals.append(interval_tuple)

    return intervals

