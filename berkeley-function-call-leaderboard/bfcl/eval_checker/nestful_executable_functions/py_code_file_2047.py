from datetime import timedelta

from typing import Union



def time_format(time: Union[float, int]) -> str:

    """Formats a time in seconds as a string with the format `HH:mm:ss.SSS`.



    Args:

        time: The time in seconds as a float or integer.



    Returns:

        The formatted time string.

    """

    milliseconds = int(time * 1000)

    seconds, milliseconds = divmod(milliseconds, 1000)

    minutes, seconds = divmod(seconds, 60)

    hours, minutes = divmod(minutes, 60)

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'

