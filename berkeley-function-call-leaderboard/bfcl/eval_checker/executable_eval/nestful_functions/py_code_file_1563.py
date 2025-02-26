from typing import Union



def convert_to_human_readable_format(minutes: int) -> str:

    """Converts a given duration in minutes to a human-readable format in hours and minutes.

    Args:

        minutes: The duration in minutes.

    Returns:

        A string in the format of `HHhMMm`, where `HH` and `MM` are the number of hours and minutes, respectively.

        If the duration is less than one hour, just the minutes will be returned (e.g., `45m`).

    """

    hours = minutes // 60

    minutes = minutes % 60

    if hours > 0:

        return f"{hours}h{minutes}m"

    else:

        return f"{minutes}m"

