import re

from typing import Union



def convert_12_to_24(time_str: str) -> str:

    """Converts a 12-hour clock time string to a 24-hour clock.



    Args:

        time_str: The input time string in 12-hour clock format.



    Returns:

        The input time string converted to 24-hour clock format.

    """

    match = re.search(r"(\d+):(\d+)\s*(am|pm)", time_str)

    if not match:

        raise ValueError("Invalid input format")

    hour, minutes, meridiem = match.groups()

    hour = int(hour)

    minutes = int(minutes)

    if meridiem == "pm" and hour != 12:

        hour += 12

    elif meridiem == "am" and hour == 12:

        hour = 0

    return f"{hour:02d}:{minutes:02d}"

