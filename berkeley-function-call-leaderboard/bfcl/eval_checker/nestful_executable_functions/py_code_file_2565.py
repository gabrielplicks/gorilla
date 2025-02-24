from typing import Any

import datetime



def get_formatted_time(time_obj: datetime.datetime) -> str:

    """Returns a string with the format `YYYY-MM-DD HH:MM:SS` based on the datetime object.



    Args:

        time_obj: The datetime object.



    Returns:

        A formatted string representing the datetime object.

    """

    year = time_obj.year

    month = time_obj.month

    day = time_obj.day

    hour = time_obj.hour

    minute = time_obj.minute

    second = time_obj.second



    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"

