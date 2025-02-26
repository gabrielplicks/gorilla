from typing import Tuple



def convert_to_hms(seconds: int) -> Tuple[int, int, int]:

    """Converts seconds to hours, minutes, and seconds.

    Args:

        seconds: The number of seconds to convert.

    Returns:

        A tuple of the form (hours, minutes, seconds).

    """

    hours = seconds // 3600

    minutes = (seconds % 3600) // 60

    seconds = (seconds % 3600) % 60

    return hours, minutes, seconds



def format_time(seconds: int) -> str:

    """Formats a time in seconds as a string in the format 'hour:minute:second'.

    Args:

        seconds: The number of seconds to format.

    Returns:

        A string in the format 'hour:minute:second'.

    """

    h, m, s = convert_to_hms(seconds)

    return f'{h}:{m}:{s}'

