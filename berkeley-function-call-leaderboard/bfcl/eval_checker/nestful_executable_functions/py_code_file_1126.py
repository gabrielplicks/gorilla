from datetime import datetime

from typing import List



def get_earliest_datetime_string(datetime_list: List[datetime]) -> str:

    """Returns a string representation of the earliest date and time in the given list of datetime objects.



    The string follows the format "YYYY-MM-DD HH:MM:SS", where YYYY represents the year, MM represents the month, DD represents the day, HH represents the hour, MM represents the minute, and SS represents the second.



    If the input list is empty or contains no valid datetime objects, the function returns an empty string.



    Args:

        datetime_list: A list of datetime objects.

    """

    if not datetime_list:

        return ""



    earliest_datetime = min(datetime_list)

    return earliest_datetime.strftime("%Y-%m-%d %H:%M:%S")

