from typing import Union



def format_time_as_str(seconds: Union[int, float]) -> str:

    """Formats a duration in seconds as a string in the format "HH:MM:SS".



    Args:

        seconds: The duration in seconds.



    Returns:

        The formatted string.

    """

    hours = seconds // 3600

    minutes = (seconds % 3600) // 60

    seconds = seconds % 60

    if hours > 0:

        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    elif minutes > 0:

        return f'{minutes:02d}:{seconds:02d}'

    else:

        return f'{seconds:02d}'

