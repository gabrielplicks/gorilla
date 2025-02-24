from datetime import datetime

from typing import Union



def timestamp_convert(ts: str) -> datetime:

    """

    Converts a timestamp string of the format "01-01-2023 12:00:00"

    to a datetime object.

    Args:

        ts: The timestamp string to be converted.

    """

    date_str = ts.split(' ')[0]

    time_str = ts.split(' ')[1]

    year = int(date_str.split('-')[2])

    month = int(date_str.split('-')[1])

    day = int(date_str.split('-')[0])

    hour = int(time_str.split(':')[0])

    min = int(time_str.split(':')[1])

    sec = int(time_str.split(':')[2])

    return datetime(year, month, day, hour, min, sec)

