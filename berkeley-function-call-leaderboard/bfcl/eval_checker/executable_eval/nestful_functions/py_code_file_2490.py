from typing import Tuple



def time_to_sec(time_string: str) -> int:

    """Converts a time string in the format `HH:MM:SS` to a number of seconds.



    Args:

        time_string: The time string to convert.



    Returns:

        The number of seconds equivalent to the input time string.

    """

    hours, minutes, seconds = time_string.split(':')

    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)



    return total_seconds

