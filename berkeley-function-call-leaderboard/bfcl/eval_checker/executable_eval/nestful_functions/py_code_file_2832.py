from datetime import datetime

from typing import Optional



def rearrange_date_string(date_string: str) -> Optional[str]:

    """Rearranges a date string into the format "yyyy/mm/dd".



    Args:

        date_string: A string representing a date in the format "yyyy/mm/dd".



    Returns:

        A string representing the date in the format "yyyy/mm/dd", or None if the input string is not a valid date.

    """

    try:

        date = datetime.strptime(date_string, "%Y/%m/%d")

        new_date_string = datetime.strftime(date, "%Y/%m/%d")

        return new_date_string

    except ValueError:

        return None

