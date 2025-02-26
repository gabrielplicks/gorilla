from typing import Tuple



def time_value_to_seconds(days: int, hours: int, minutes: int, seconds: int) -> int:

    """Converts a time value expressed in days, hours, minutes, and seconds into the corresponding value expressed in seconds.



    Args:

        days: The number of days.

        hours: The number of hours.

        minutes: The number of minutes.

        seconds: The number of seconds.



    Returns:

        The corresponding time value in seconds.

    """

    seconds_in_day = 86400

    seconds_in_hour = 3600

    seconds_in_minute = 60



    total_seconds = days * seconds_in_day + hours * seconds_in_hour + minutes * seconds_in_minute + seconds



    return total_seconds

