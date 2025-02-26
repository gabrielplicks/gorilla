from typing import Union



def calculate_total_seconds(hours: Union[int, float], minutes: Union[int, float], seconds: Union[int, float]) -> int:

    """Calculates the total number of seconds in a given number of hours, minutes, and seconds.



    Args:

        hours: The number of hours.

        minutes: The number of minutes.

        seconds: The number of seconds.



    Returns:

        The total number of seconds.

    """

    total = hours * 3600 + minutes * 60 + seconds

    return total

