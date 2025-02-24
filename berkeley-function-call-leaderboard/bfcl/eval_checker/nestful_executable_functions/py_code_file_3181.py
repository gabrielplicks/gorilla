from typing import List



def date_format(s: str) -> str:

    """Formats a date string from "YYYY-MM-DD" to "MM-DD-YYYY".



    Args:

        s: The input date string in the format "YYYY-MM-DD".



    Returns:

        The formatted date string in the format "MM-DD-YYYY".

    """

    year, month, day = s.split('-')

    return '-'.join([month, day, year])

