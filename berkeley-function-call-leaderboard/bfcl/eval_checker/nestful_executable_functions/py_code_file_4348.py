from typing import AnyStr



def string_is_alphanumeric(input_string: AnyStr) -> bool:

    """Check if an input string contains only alphanumeric characters.



    Args:

        input_string: The input string to check.



    Returns:

        True if the string contains only alphanumeric characters, False otherwise.

    """

    return all(ch.isalnum() for ch in input_string)

