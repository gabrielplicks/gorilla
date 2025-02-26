from datetime import datetime

from typing import Union



def compare_timestamps(timestamp1: Union[str, datetime], timestamp2: Union[str, datetime]) -> bool:

    """Compares the value of two timestamps and returns a boolean value.



    Args:

        timestamp1: The first timestamp.

        timestamp2: The second timestamp.



    Returns:

        True if the value of the second timestamp is greater than the first, False otherwise.

    """

    epoch1 = datetime.strptime(timestamp1, '%Y-%m-%d %H:%M:%S').strftime('%s')

    epoch2 = datetime.strptime(timestamp2, '%Y-%m-%d %H:%M:%S').strftime('%s')

    epoch1 = int(epoch1)

    epoch2 = int(epoch2)

    time_diff = epoch2 - epoch1

    return time_diff > 0

