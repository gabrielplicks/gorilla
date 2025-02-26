from typing import Set



def verify_boolean(s: str) -> bool:

    """Determines whether a string represents a boolean value.



    Args:

        s: The string to be checked.



    Returns:

        True if the string represents a valid boolean value, False otherwise.

    """

    valid_bools: Set[str] = {'true', 'false', '1', '0'}

    return s.lower() in valid_bools

