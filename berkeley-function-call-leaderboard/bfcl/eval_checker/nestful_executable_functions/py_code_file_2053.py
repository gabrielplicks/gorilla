import re

import keyword



def is_valid_class_name(name: str) -> bool:

    """Checks if a string is a valid class name in Python.



    Args:

        name: The string to check.



    Returns:

        True if the string is a valid class name, False otherwise.

    """

    if keyword.iskeyword(name):

        return False

    if re.search(r'^[0-9]', name):

        return False

    if " " in name:

        return False

    if re.search(r'[^a-zA-Z0-9_]', name):

        return False

    return True

