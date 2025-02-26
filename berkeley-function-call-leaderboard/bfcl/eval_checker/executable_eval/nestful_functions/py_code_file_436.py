import time

from typing import Union



def generate_node_name(unix_timestamp: Union[int, float]) -> str:

    """Generates a node name based on the current time in the HH:MM:SS format.



    Args:

        unix_timestamp: The current time as a Unix timestamp (seconds since epoch).

    """

    dt = time.gmtime(unix_timestamp)

    node_name = time.strftime("%H:%M:%S", dt)



    return node_name

