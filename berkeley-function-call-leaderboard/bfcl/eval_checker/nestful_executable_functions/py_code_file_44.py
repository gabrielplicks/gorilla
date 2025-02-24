from datetime import datetime

from typing import Union



def is_weekend(date: Union[str, datetime]) -> bool:

    """Determines whether a given `date` string is a weekend, which is defined as a Saturday or Sunday.

    The `date` string is a string of the form "YYYY-MM-DD" in ISO 8601 format.



    Args:

        date: The date to check.

    """

    if isinstance(date, str):

        date_object = datetime.strptime(date, '%Y-%m-%d')

    else:

        date_object = date

    return date_object.weekday() in [5, 6]  # 5 is Saturday, 6 is Sunday

