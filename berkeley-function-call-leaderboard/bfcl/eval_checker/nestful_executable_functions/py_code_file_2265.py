import re



def is_numeric_string(input_string: str) -> bool:

    """Determines whether a given string is a valid numeric string.



    Args:

        input_string: The string to check.



    Returns:

        True if the input string is a valid numeric string, False otherwise.

    """

    try:

        if re.match(r'^[+-]?\d+$|^[+-]?\d*\.?\d+$', input_string):

            return True

        else:

            return False

    except Exception:

        return False

