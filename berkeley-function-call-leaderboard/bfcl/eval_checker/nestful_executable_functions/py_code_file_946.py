from dateutil import parser

from dateutil.tz import tzutc

from typing import Union



def strip_timezone_info(date_str: str) -> str:

    """Removes the timezone information in the given date string.

    Args:

        date_str: The date string to remove the timezone information from.

    Returns:

        The string without the timezone information.

    Raises:

        ValueError: If the timezone information is missing.

    """

    try:

        parsed_date = parser.parse(date_str)

    except ValueError as e:

        raise ValueError(f"Invalid date format: {date_str}")



    stripped_date = parsed_date.replace(tzinfo=None)

    return stripped_date.strftime("%Y-%m-%dT%H:%M:%S")

