from typing import Tuple



def format_date_string(date_string: str) -> Tuple[int, int, int]:

    """Formats a string of the form "01/01/2023" into a tuple of integers (day, month, year).



    Args:

        date_string: A string of the form "01/01/2023".



    Returns:

        A tuple of integers (day, month, year).

    """

    parts = date_string.split('/')

    day = int(parts[0])

    month = int(parts[1])

    year = int(parts[2])

    return day, month, year

