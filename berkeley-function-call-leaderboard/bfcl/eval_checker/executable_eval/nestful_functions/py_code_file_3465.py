from typing import Union

import datetime



def date_validation_with_graceful_error_handling(date_str: str) -> Union[str, datetime.date]:

    """Checks if a given date string is a valid date format and adds 10 years to it.



    Args:

        date_str: The date string to check.



    Returns:

        The new date string in the format `YYYY-MM-DD` if the input is a valid date.

        Otherwise, returns `"Invalid date format"`.

    """

    try:

        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")

        new_date = date.replace(year=date.year + 10) - datetime.timedelta(days=5)

        return new_date.strftime("%Y-%m-%d")

    except ValueError:

        return "Invalid date format"

