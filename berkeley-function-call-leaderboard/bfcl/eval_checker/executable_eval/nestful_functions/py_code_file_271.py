from typing import Union



def round_to_nearest(num: Union[int, float], multiple: Union[int, float]) -> Union[int, float]:

    """Rounds a number to the nearest multiple of another given number.



    Args:

        num: The number to be rounded.

        multiple: The multiple to which the number is to be rounded.

    """

    return round(num / multiple) * multiple

