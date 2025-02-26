from datetime import datetime

from typing import Optional



def get_formatted_timestamp(dt: Optional[datetime] = None) -> str:

    """

    Returns a formatted timestamp string. If no datetime is provided, the current time is used.

    The timestamp follows the format "YYYY/MM/DD HH:MM:SS".

    Args:

        dt: The datetime object to format. If None, the current time is used.

    Raises:

        ValueError: If the input datetime is invalid.

    """

    if dt is None:

        dt = datetime.now()



    if not isinstance(dt, datetime):

        raise ValueError("Invalid datetime provided")



    return dt.strftime("%Y/%m/%d %H:%M:%S")

