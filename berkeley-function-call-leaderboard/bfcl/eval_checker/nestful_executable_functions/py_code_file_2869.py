from typing import Tuple



def time_span_to_seconds(time_span: str) -> int:

    """Converts a string representing a time span into a total number of seconds.



    Args:

        time_span: A string representing a time span in the format 'Xh Ym Zs', where X, Y, and Z are positive integers.



    Returns:

        The total number of seconds.

    """

    components = time_span.split()

    hours = int(components[0][:-1])  # Remove the 'h' suffix

    minutes = int(components[1][:-1])  # Remove the 'm' suffix

    seconds = int(components[2][:-1])  # Remove the 's' suffix

    total_seconds = hours * 60 * 60 + minutes * 60 + seconds



    return total_seconds

