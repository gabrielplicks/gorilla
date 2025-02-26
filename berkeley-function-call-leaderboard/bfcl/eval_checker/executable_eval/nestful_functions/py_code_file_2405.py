from typing import Union



def format_pretty_time(duration_seconds: Union[int, float]) -> str:

    """Formats a duration in seconds into a string representing hours, minutes, and seconds.

    Args:

        duration_seconds: The duration in seconds.

    Returns:

        A formatted string representing the duration in hours, minutes, and seconds.

    """

    hours = duration_seconds // 3600

    remaining_seconds = duration_seconds % 3600

    minutes = remaining_seconds // 60

    seconds = remaining_seconds % 60

    return f"{hours:02}:{minutes:02}:{seconds:02}"

