import re

import typing



def is_valid_filename(filename: str) -> bool:

    """Checks if a given string is a valid file name by comparing it against the pattern `filename-????.py` where `?` is a placeholder for any single character.



    Args:

        filename: The string to check.



    Returns:

        True if the given string matches the pattern, False otherwise.

    """

    regex = re.compile(r'^filename-\w\w\w\w.py$')

    match = regex.search(filename)

    if match:

        return True

    else:

        return False

