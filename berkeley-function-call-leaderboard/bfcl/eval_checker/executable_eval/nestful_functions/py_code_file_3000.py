from typing import Union



def duration_to_string(duration: Union[int, float]) -> str:

    """Converts a time duration in seconds to a string of the form "HH:MM:SS".

    Args:

        duration: A positive integer or float representing a duration in seconds.

    Raises:

        ValueError: If the input is negative or 0.

    """

    if duration <= 0:

        raise ValueError("Duration must be positive")



    hours = duration // 3600  # Compute hours

    minutes = (duration % 3600) // 60  # Compute minutes

    seconds = duration % 60  # Compute seconds

    return f"{hours:02}:{minutes:02}:{seconds:02}"

