import time

from typing import Tuple



def seconds_from_epoch(struct_time: Tuple[int, int, int, int, int, int, int, int, int]) -> int:

    """Calculates the number of seconds from the epoch (Jan 1, 1970) to the input time.



    Args:

        struct_time: The input time as a `struct_time` object.



    Returns:

        The number of seconds from the epoch to the input time.

    """

    epoch_seconds = time.mktime(time.gmtime(0))

    input_seconds = time.mktime(struct_time)

    return input_seconds - epoch_seconds

