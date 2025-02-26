from datetime import datetime

from typing import Union



def ensure_leading_zero(date_string: Union[str, bytes]) -> str:

    """

    Takes a string representing a date in the format `YYYY-MM-DD` and returns a string of the same date with a leading zero added to the day if necessary, as in `YYYY-MM-DD`.



    Args:

        date_string: A string representing a date in the format `YYYY-MM-DD`.

    """

    date = datetime.strptime(date_string, "%Y-%m-%d")

    return date.strftime("%Y-%m-%d")

