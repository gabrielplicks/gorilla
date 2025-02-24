from datetime import datetime

from typing import List, Optional



def find_earliest_and_latest(times: List[datetime]) -> Optional[List[datetime]]:

    """

    Finds the earliest and latest datetimes from a list of datetime objects.

    If the input list is empty, the function returns None.



    Args:

        times: A list of datetime objects.



    Returns:

        A list containing the earliest and latest datetimes from the input list,

        or None if the input list is empty.

    """

    if times:

        earliest = min(times)

        latest = max(times)

        return [earliest, latest]

    else:

        return None

