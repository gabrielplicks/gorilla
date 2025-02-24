from typing import Union



def str2bool(value: Union[str, bool]) -> bool:

    """Converts a string value to a boolean.

    Args:

        value: The string value to convert.

    Returns:

        The boolean value of the input string.

    Raises:

        TypeError: If the input is not a string.

        ValueError: If the input string does not match any of the expected boolean patterns.

    """

    if not isinstance(value, str):

        raise TypeError("Input must be a string")

    value = value.lower()

    if value in ["true", "yes", "y", "t", "1"]:

        return True

    elif value in ["false", "no", "n", "f", "0"]:

        return False

    else:

        raise ValueError("Unexpected boolean value: {}".format(value))

