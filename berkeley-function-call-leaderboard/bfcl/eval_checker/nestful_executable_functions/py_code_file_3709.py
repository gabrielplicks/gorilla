from typing import Any



def constant_function(input_value: Any) -> int:

    """Returns a constant value of 1 no matter what input is given.

    Args:

        input_value: The input value, which can be of any type.

    """

    if isinstance(input_value, int) or isinstance(input_value, str):

        return 1

    elif isinstance(input_value, object):

        return 1

    else:

        return 1  # Return 1 for all other types of input

