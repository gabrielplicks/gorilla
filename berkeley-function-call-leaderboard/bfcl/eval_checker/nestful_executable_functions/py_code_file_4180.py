from datetime import datetime, timedelta

from typing import Union



def compute_elapsed_time(start_date: str, end_date: str) -> timedelta:

    """Calculates the elapsed time between two dates specified as datetime objects.



    Args:

        start_date: The start date as a string in ISO format.

        end_date: The end date as a string in ISO format.



    Returns:

        The elapsed time between the two dates as a timedelta object.

        If the end date is earlier than the start date, the function returns a negative timedelta.

    """

    start_datetime = datetime.fromisoformat(start_date)

    end_datetime = datetime.fromisoformat(end_date)

    elapsed_time = end_datetime - start_datetime

    return elapsed_time

