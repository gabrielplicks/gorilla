import re

import typing



def check_phone_numbers(filename: str) -> bool:

    """Checks if each line in the file matches a regular expression for a phone number.



    Args:

        filename: The name of the file to check.



    Returns:

        True if a line matches the regular expression, False otherwise.

    """

    with open(filename, 'r') as file:

        for line in file:

            match = re.match(r'^\+?(\d{1,3})-(\d{3})-(\d{3,10})$', line)

            if match:

                return True

        return False

