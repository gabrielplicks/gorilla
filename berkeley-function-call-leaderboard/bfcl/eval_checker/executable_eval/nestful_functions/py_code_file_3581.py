import re

import typing



def is_date(s: str) -> bool:

    """Checks if a string matches the pattern 'YYYY-MM-DD'.



    Args:

        s: The string to check.



    Returns:

        True if the string matches the pattern, False otherwise.

    """

    pattern = r'\d{4}-\d{2}-\d{2}'

    return bool(re.search(pattern, s))

