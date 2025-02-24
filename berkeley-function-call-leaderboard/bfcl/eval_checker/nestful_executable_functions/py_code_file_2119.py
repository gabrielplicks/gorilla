from typing import Union



def convert_to_duration_str(seconds: Union[int, float]) -> str:

    """Converts a time duration in seconds into a string representation of hours, minutes, and seconds.

    Args:

        seconds: The time duration in seconds.

    Returns:

        A string in the format "HH:MM:SS", where HH, MM, and SS represent the hours, minutes, and seconds, respectively.

        If the input is an invalid number, returns "Invalid duration".

    """

    if not isinstance(seconds, (int, float)) or seconds < 0:

        return "Invalid duration"

    hours = seconds // 3600

    minutes = (seconds % 3600) // 60

    seconds = seconds % 60

    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return duration_str

