from typing import Union



def format_time_delta(delta: int) -> str:

    """Formats a time delta in seconds into a human-readable string.



    Args:

        delta: The time delta in seconds.



    Returns:

        A string in the format `{H}h {M}m {S}s`, where `{H}` is the number of hours,

        `{M}` is the number of minutes, and `{S}` is the number of seconds.

    """

    hours = delta // 3600

    minutes = (delta % 3600) // 60

    seconds = (delta % 3600) % 60

    return f"{hours}h {minutes}m {seconds}s"

