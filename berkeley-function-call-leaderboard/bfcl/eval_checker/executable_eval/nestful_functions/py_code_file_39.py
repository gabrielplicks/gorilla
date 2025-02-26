from typing import Tuple

import datetime



def timestamp_to_str(timestamp: Tuple[int, int, int, int, int, int]) -> str:

    """Converts a Unix timestamp (seconds since epoch) to a human-readable string with the format "%Y-%m-%d %H:%M:%S".



    Args:

        timestamp: A tuple of integers representing the year, month, day, hour, minute, and second.



    Returns:

        A string representing the timestamp in the format "%Y-%m-%d %H:%M:%S".

    """

    year, month, day, hour, minute, second = timestamp

    dt = datetime.datetime(year, month, day, hour, minute, second)

    return dt.strftime("%Y-%m-%d %H:%M:%S")

