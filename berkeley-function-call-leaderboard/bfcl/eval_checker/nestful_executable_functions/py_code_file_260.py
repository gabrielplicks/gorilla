import re



def is_valid_name(s: str) -> bool:

    """Checks if a given string is a valid name.



    A valid name is defined as a string that does not contain any non-alphabetic characters and has a length of at least 2.



    Args:

        s: The string to be checked.



    Returns:

        True if the string is a valid name, False otherwise.

    """

    return re.match(r"^[a-zA-Z]{2,}$", s) is not None

