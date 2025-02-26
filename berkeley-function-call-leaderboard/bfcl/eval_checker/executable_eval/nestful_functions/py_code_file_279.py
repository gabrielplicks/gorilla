from datetime import datetime, timedelta

from typing import Union



def previous_weekday(date_string: str, weekday: str) -> str:

    """

    Returns the date of the previous weekday as a string with the same format.

    Args:

        date_string: A string representing a date in the format %Y-%m-%d.

        weekday: A string representing a weekday in English.

    """

    date = datetime.strptime(date_string, "%Y-%m-%d")

    weekday_index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(weekday)

    if date.weekday() != weekday_index:

        date -= timedelta(days=date.weekday() - weekday_index)

    return date.strftime("%Y-%m-%d")

