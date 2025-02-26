from datetime import datetime

from typing import Union



def days_since(timestamp: Union[int, float]) -> int:

    """Calculates the number of days between a given date and the current date in the local time zone.

    Args:

        timestamp: The timestamp in milliseconds.

    """

    given_date = datetime.fromtimestamp(timestamp / 1000)

    current_date = datetime.now()

    days_passed = (current_date - given_date).days

    return days_passed

