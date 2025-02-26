from typing import Any

import datetime



def get_current_date_time_iso8601() -> str:

    """Generates a string containing the current date and time in the ISO 8601 format.



    Args:

        None



    Returns:

        A string containing the current date and time in the ISO 8601 format.

    """

    now = datetime.datetime.now()

    return f"{now.year:04}-{now.month:02}-{now.day:02}T{now.hour:02}:{now.minute:02}:{now.second:02}"

