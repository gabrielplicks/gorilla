from typing import Union

import datetime



def get_formatted_datetime_string(dt: Union[datetime.datetime, datetime.date]) -> str:

    """Formats a datetime object to a string of the form 'yyyy-mm-dd hh:mm:ssZ'.



    Args:

        dt: The datetime object to be formatted.



    Returns:

        A string representation of the datetime object in the format 'yyyy-mm-dd hh:mm:ssZ'.

    """

    dt_without_timezone = dt.replace(tzinfo=None)

    return dt_without_timezone.strftime("%Y-%m-%d %H:%M:%S") + "Z"

