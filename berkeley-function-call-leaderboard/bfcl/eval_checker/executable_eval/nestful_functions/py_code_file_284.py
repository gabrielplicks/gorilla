from datetime import datetime

from typing import Union



def quote_date(date: Union[datetime, str]) -> str:

    """Returns a string representation of the date in the format: "{year}-{month}-{day}".



    Args:

        date: A Python datetime object or a string representation of a date.

    """

    if isinstance(date, str):

        date = datetime.strptime(date, "%Y-%m-%d")

    return date.strftime("%Y-%m-%d")

