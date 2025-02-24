from datetime import datetime

from typing import Union



def get_time_string(time: Union[datetime, None]) -> str:

    """Creates a time string in the format "MM/DD/YYYY HH:MM:SS AM/PM".



    Args:

        time: A datetime object.

    """

    return time.strftime("%m/%d/%Y %I:%M:%S %p")

