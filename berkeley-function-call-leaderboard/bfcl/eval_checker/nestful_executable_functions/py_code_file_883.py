from typing import Union



def max_none(a: Union[int, None], b: Union[int, None]) -> Union[int, None]:

    """Returns the maximum of two integers a and b. If both are None, returns None.

    If one of the integers is None, returns the other integer.

    Args:

        a: The first integer.

        b: The second integer.

    """

    if a is None and b is None:

        return None

    elif a is None:

        return b

    elif b is None:

        return a

    else:

        return max(a, b)

