from typing import Union



def is_positive_integer(value: Union[int, str]) -> bool:

    """Checks if a value is a positive integer or a string representation of a positive integer.



    Args:

        value: The value to check.



    Returns:

        True if the value is a positive integer or a string representation of a positive integer, False otherwise.

    """

    if not isinstance(value, str):

        value = str(value)

    return value.isdigit() and value[0] != "0"

