from typing import Union



def format_time_duration(duration: Union[int, float]) -> str:

    """Formats an input time duration (in seconds) into a human-readable string.



    Args:

        duration: The duration in seconds.



    Returns:

        A human-readable string representing the duration.

    """

    hours = duration // 3600

    minutes = (duration % 3600) // 60

    seconds = duration % 60

    if hours > 0:

        formatted_duration = f"{hours}h{minutes:02}m{seconds:02}s"

    elif minutes > 0:

        formatted_duration = f"{minutes}m{seconds:02}s"

    else:

        formatted_duration = f"{seconds}s"



    return formatted_duration

