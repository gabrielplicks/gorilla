from typing import Union



def time_format_convert_d2h(seconds: Union[int, float]) -> str:

    """Converts a number of seconds to a string representing the equivalent number of days, hours, minutes, and seconds.

    Args:

        seconds: The number of seconds to convert.

    """

    days = seconds // 86400

    remainder_seconds = seconds % 86400

    hours = remainder_seconds // 3600

    remainder_seconds %= 3600

    minutes = remainder_seconds // 60

    seconds = remainder_seconds % 60

    return f"{days}:{hours}:{minutes}:{seconds}"

