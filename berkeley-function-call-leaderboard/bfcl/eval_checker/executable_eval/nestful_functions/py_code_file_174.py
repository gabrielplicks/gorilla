from datetime import datetime

from typing import Union



def date_to_unix_timestamp(date_string: str) -> Union[int, float]:

    """Converts a date string in the format MM/DD/YYYY to a Unix timestamp.



    Args:

        date_string: The date string in the format MM/DD/YYYY.



    Returns:

        The Unix timestamp corresponding to the given date string.

    """

    date_obj = datetime.strptime(date_string, '%m/%d/%Y')

    unix_timestamp = date_obj.timestamp()



    return unix_timestamp

