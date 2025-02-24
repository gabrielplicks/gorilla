import time

from typing import List



def sleep_and_return(items: List[int], sleep_duration: float) -> List[int]:

    """Sleeps for the specified duration before returning the list.



    Args:

        items: A list of integers.

        sleep_duration: The duration to sleep in seconds.

    """

    time.sleep(sleep_duration)

    return items

