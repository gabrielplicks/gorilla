from datetime import datetime

from typing import List



def count_days(date1: List[int], date2: List[int]) -> int:

    """Computes the number of days between two dates.

    Args:

        date1: A list representing the year, month, and day of a date.

        date2: A list representing the year, month, and day of a date.

    """

    date1_dt = datetime(date1[0], date1[1], date1[2])

    date2_dt = datetime(date2[0], date2[1], date2[2])

    return abs((date2_dt - date1_dt).days)

