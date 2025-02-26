from typing import Tuple



def time_format_conversion(time_string: str) -> str:

    """Converts a time string from the format `HH:MM:SS` to `D:HH:MM:SS`.



    Args:

        time_string: The time string to be converted.



    Returns:

        The converted time string.

    """

    hours, minutes, seconds = time_string.split(":")

    total_days = int(hours) // 24

    remaining_hours = int(hours) % 24

    return f"{total_days}:{remaining_hours}:{minutes}:{seconds}"

