import re



def check_if_valid_number(string: str) -> bool:

    """Checks if a string is a valid number.



    Args:

        string: The string to check.



    Returns:

        True if the string is a valid number, False otherwise.

    """

    pattern = r'^[-+]?[0-9]+(\.[0-9]+)?$'

    return bool(re.match(pattern, string))

