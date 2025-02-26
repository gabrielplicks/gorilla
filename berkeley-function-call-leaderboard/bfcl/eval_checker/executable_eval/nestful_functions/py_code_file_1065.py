import re

import typing



def check_if_valid_password(password: str) -> bool:

    """Checks if a password string satisfies certain criteria.

    Args:

        password: A string representing the password.

    Returns:

        True if the password follows the criteria, False otherwise.

    """

    pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{6,16}$')

    if pattern.match(password):

        return True

    else:

        return False

