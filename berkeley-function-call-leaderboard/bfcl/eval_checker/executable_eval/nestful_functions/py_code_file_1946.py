import re



def escape_special_characters(input_string: str) -> str:

    """Escapes all special characters in a given string using Python's `re.escape` method.



    Args:

        input_string: The input string to escape special characters in.



    Returns:

        The input string with all special characters escaped.

    """

    return re.escape(input_string)

