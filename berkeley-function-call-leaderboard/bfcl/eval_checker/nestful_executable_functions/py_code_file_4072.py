from typing import Tuple



def extract_hours_minutes_from_seconds(seconds: int) -> Tuple[int, int]:

    """Extracts hours and minutes from a given number of seconds.



    Args:

        seconds: The given number of seconds.



    Returns:

        A tuple of the hours and minutes as integers.

    """

    hours = seconds // 3600

    remaining_seconds = seconds % 3600

    minutes = remaining_seconds // 60



    return hours, minutes

