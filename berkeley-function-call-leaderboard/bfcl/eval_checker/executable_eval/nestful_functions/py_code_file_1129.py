from datetime import datetime, timezone, timedelta

from typing import Union



def get_next_datetime(current_datetime: datetime, day_of_week: str) -> datetime:

    """

    Returns a datetime object representing the date and time for the next given day of the week.



    Args:

        current_datetime: The current datetime object.

        day_of_week: The day of the week.

    """

    current_datetime = current_datetime.astimezone(timezone.utc)

    day_of_week_idx = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week)

    days_to_add = (day_of_week_idx - current_datetime.weekday()) % 7

    return current_datetime + timedelta(days=days_to_add)

