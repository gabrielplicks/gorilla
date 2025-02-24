from datetime import datetime

from typing import Union



def milliseconds_since(time: Union[datetime, str]) -> int:

    """Calculates the number of milliseconds elapsed since a given datetime object.

    Args:

        time: The datetime object or string representation of the datetime object.

    """

    if isinstance(time, str):

        time = datetime.fromisoformat(time)

    return (time - datetime.fromtimestamp(0)).total_seconds() * 1000

