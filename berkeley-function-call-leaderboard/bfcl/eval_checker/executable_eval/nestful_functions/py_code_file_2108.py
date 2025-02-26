import re



def starts_with_valid_identifier(string: str) -> bool:

    """Checks if a string starts with a valid Python identifier.



    Args:

        string: The string to check.



    Returns:

        True if the string starts with a valid Python identifier, otherwise False.

    """

    return re.match(r'^[a-zA-Z_]\w*$', string) is not None

