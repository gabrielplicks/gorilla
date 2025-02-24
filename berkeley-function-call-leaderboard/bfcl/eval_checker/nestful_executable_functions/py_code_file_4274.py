from typing import Union



def is_valid_hour(hour: Union[int, None]) -> Union[bool, None]:

    """Checks if the given hour is a valid hour in the 24-hour format (i.e., between 0 and 23 inclusive).

    Args:

        hour: The hour to check.

    """

    if not isinstance(hour, int):

        return None

    if hour not in range(24):

        return None

    return True

