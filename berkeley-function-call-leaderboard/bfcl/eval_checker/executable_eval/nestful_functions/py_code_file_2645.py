from typing import Union



def time_string(seconds: Union[int, float]) -> str:

    """Returns a string representation of the time in the format "{H}:{M}:{S}", where "H", "M", and "S" are the number of hours, minutes, and seconds, respectively.



    Args:

        seconds: The number of seconds.

    """

    hours = seconds // 3600

    minutes = (seconds % 3600) // 60

    seconds = (seconds % 3600) % 60

    return f"{hours}:{minutes}:{seconds}"

