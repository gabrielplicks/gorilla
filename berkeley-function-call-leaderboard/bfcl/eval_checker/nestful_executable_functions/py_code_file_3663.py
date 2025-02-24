import re



def validate_int_string(s: str) -> bool:

    """Validates if a string represents an integer.



    Args:

        s: The string to validate.



    Returns:

        True if the string represents an integer, False otherwise.

    """

    return bool(re.match(r"^[+-]?\d+$", s))

