import re



def replace_multiple_spaces(input_string: str) -> str:

    """Replaces all contiguous whitespace in a string with a single space.



    Args:

        input_string: The input string.



    Raises:

        TypeError: If the input is not a string.

    """

    try:

        return re.sub(r'\s+', ' ', input_string)

    except TypeError:

        raise TypeError("Invalid input. Input must be a string.")

