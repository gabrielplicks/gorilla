from typing import Union



def round_to_two_decimals(num: Union[int, float]) -> str:

    """

    Rounds a number to two decimal places and returns the result as a string.

    Args:

        num: The number to be rounded.

    """

    try:

        rounded_num = round(num, 2)

        return format(rounded_num, '.2f')

    except TypeError:

        return 'Invalid input'

