from typing import Union

import datetime



def validate_and_format_time(time_str: str) -> Union[str, ValueError]:

    """

    Validates and formats a 24-hour time string to a 12-hour time string with AM or PM.



    Args:

        time_str: A string representing a 24-hour time.



    Raises:

        ValueError: If the input time string is not a valid 24-hour time.



    Returns:

        A string representing the reformatted 12-hour time with AM or PM.

    """

    try:

        time = datetime.datetime.strptime(time_str, "%H:%M")

        formatted_time = time.strftime("%-I:%M %p")

        return formatted_time

    except ValueError:

        raise ValueError("Invalid time format")

