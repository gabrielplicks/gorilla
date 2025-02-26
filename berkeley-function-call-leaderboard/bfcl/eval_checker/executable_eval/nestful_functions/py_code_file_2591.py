import datetime

from typing import Union



def add_seconds(time_string: str, seconds_to_add: Union[int, float]) -> str:

    """Adds a specified number of seconds to a given time string and returns the result.



    Args:

        time_string: The input time string in the format of "HH:MM:SS".

        seconds_to_add: The number of seconds to add.

    """

    time_parts = time_string.split(":")

    hours = int(time_parts[0])

    minutes = int(time_parts[1])

    seconds = int(time_parts[2])

    time = datetime.datetime(1, 1, 1, hours, minutes, seconds)

    time += datetime.timedelta(seconds=seconds_to_add)

    return time.strftime("%H:%M:%S")

