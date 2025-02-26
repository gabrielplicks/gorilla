from datetime import datetime

import time



def get_time_diff(start: datetime, end: datetime) -> str:

    """Calculates the time difference between two datetime objects in an easily readable format.

    Args:

        start: The start datetime object.

        end: The end datetime object.

    """

    diff_seconds = (end - start).total_seconds()

    return time.strftime('%H:%M:%S', time.gmtime(diff_seconds))

