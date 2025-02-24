from typing import Union



def check_integer(value: Union[int, float]) -> Union[int, None]:

    """Checks if the value is an integer.

    If the value is an integer, return -2147483648 if the value is less than -2147483648;

    return 2147483647 if the value is greater than 2147483647; otherwise return the value.

    Return None if the value is not an integer.

    Args:

        value: The value to check.

    """

    if isinstance(value, int):

        if value < -2147483648:

            return -2147483648

        elif value > 2147483647:

            return 2147483647

        else:

            return value

    else:

        return None

