from typing import Union



def time_remaining(timeout: int) -> Union[tuple[int, int], str]:

    """Calculates the remaining time until a timeout is complete.



    Args:

        timeout: The timeout in seconds.



    Returns:

        A tuple of (minutes, seconds) or a string that indicates that the timeout has expired.

    """

    minutes, seconds = divmod(timeout, 60)

    if timeout > 0:

        return (minutes, seconds)

    elif timeout == 0:

        return (0, 0)

    else:

        return "The timeout has already expired."

