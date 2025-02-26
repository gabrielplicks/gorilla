from typing import Optional

from datetime import datetime

from pytz import utc



def convert_local_to_utc(local_timestamp: datetime) -> Optional[datetime]:

    """Converts a timestamp from the local timezone to the UTC timezone.

    Handles any conversion errors and returns None when an exception occurs.

    Args:

        local_timestamp: The timestamp in the local timezone.

    """

    try:

        utc_timestamp = local_timestamp.astimezone(utc)

        return utc_timestamp

    except Exception:

        return None

