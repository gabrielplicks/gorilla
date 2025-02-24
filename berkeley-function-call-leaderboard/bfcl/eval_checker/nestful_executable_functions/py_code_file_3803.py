from typing import Union



def min_with_default(a: int, b: int, default: Union[int, None] = None) -> int:

    """

    Returns the smaller of two integer values. If the two values are equal,

    returns the smaller of the two values by default. However, the function

    also supports accepting an optional `default` parameter to override the

    default behavior and return the default value when the two values are equal.



    Args:

        a: The first integer value.

        b: The second integer value.

        default: The default value to return when the two values are equal.

            If not provided, the function will return the smaller of the two

            values by default.

    """

    if a < b:

        return a

    elif b < a:

        return b

    else:

        if default is not None:

            return default

        else:

            return a

