from typing import Union

from datetime import timedelta



def timedelta_to_seconds(td: Union[timedelta, None]) -> float:

    """Calculates the number of seconds from a timedelta object.



    Args:

        td: The timedelta object to convert to seconds.



    Returns:

        The number of seconds as a float.

    """

    if td is None:

        return 0

    return (td.days * 24 * 60 * 60) + (td.seconds) + (td.microseconds / 1000000)

