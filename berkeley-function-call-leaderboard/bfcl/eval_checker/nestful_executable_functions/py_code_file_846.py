from math import ceil

from typing import Union



def round_to_multiple(number: Union[int, float], multiple: Union[int, float]) -> Union[int, float]:

    """Rounds up a given number to the nearest multiple of another number.



    Args:

        number: The number to round up.

        multiple: The multiple to round up to.

    """

    if multiple < 0:

        return -1 * ceil(abs(number) / abs(multiple)) * abs(multiple)

    return ceil(number / multiple) * multiple

