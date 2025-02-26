from datetime import datetime

from typing import Union



def format_datetime_object(dt: Union[datetime, str]) -> str:

    """Formats a datetime object as a string using the format string '%Y-%m-%d-%H-%M'.



    Args:

        dt: The datetime object to format.

    """

    return dt.strftime('%Y-%m-%d-%H-%M')

