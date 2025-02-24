from datetime import datetime

from typing import Union



def get_weekday_from_date(date_str: str) -> Union[int, ValueError]:

    """

    Returns the weekday as an integer (0 for Monday, 1 for Tuesday, ..., 6 for Sunday) for a given date string in the format of "%Y-%m-%d".

    If the input date is not in the correct format, the function raises a ValueError.



    Args:

        date_str: The date string in the format of "%Y-%m-%d".

    """

    try:

        date = datetime.strptime(date_str, "%Y-%m-%d")

        return date.weekday()

    except ValueError:

        raise ValueError("Invalid date format: the date should be provided in the format of '%Y-%m-%d'")

