from typing import Union



def format_time_interval(interval: Union[int, float]) -> str:

    """Formats a given time interval into a string in the format `HH:MM:SS`.



    Args:

        interval: The time interval in seconds.



    Returns:

        A string representation of the time interval in the format `HH:MM:SS`.

    """

    hours = interval // 3600  # Divide the interval by the number of seconds in an hour

    minutes = (interval % 3600) // 60  # Calculate the remaining minutes

    seconds = interval % 60  # Calculate the remaining seconds



    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

