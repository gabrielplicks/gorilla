from datetime import datetime

from typing import Union



def get_time_diff_in_min(date1: str, date2: str) -> float:

    """Calculates the time difference in minutes between two given dates.



    Args:

        date1: A string representing the first date in the format '%Y-%m-%d %H:%M:%S'.

        date2: A string representing the second date in the format '%Y-%m-%d %H:%M:%S'.



    Returns:

        The time difference in minutes between the two dates.

    """

    date1_dt = datetime.strptime(date1, '%Y-%m-%d %H:%M:%S')

    date2_dt = datetime.strptime(date2, '%Y-%m-%d %H:%M:%S')

    time_diff = (date2_dt - date1_dt).total_seconds() / 60



    return time_diff

