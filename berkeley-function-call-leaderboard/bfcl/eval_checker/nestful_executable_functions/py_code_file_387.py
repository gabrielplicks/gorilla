from datetime import datetime

from typing import List



def get_some_datetime_strings(dt: datetime) -> List[str]:

    """Constructs a list of strings representing the date and time in different formats.



    Args:

        dt: A datetime object.



    Returns:

        A list of strings representing the date and time in different formats.

    """

    return [

        dt.strftime("%Y-%m-%d %H:%M:%S"),

        dt.strftime("%Y-%m-%d"),

        dt.strftime("%H:%M:%S"),

        dt.strftime("%d-%m-%Y %H:%M:%S"),

    ]

