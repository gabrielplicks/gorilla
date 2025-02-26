from typing import Union



def boolean_to_string(value: Union[bool, type(None)]) -> str:

    """Converts a boolean value or NoneType object to its corresponding string representation.



    Args:

        value: The boolean value or NoneType object to convert.



    Returns:

        The string representation of the input value.

    """

    return "true" if value is True else ("false" if value is False else "null")

