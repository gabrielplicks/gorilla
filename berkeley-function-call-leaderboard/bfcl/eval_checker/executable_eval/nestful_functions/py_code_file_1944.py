import re

import typing



def column_title_to_number(column_title: str) -> int:

    """Converts a column title from Excel Sheets to a number.



    Args:

        column_title: The column title to convert.



    Raises:

        AssertionError: If the input contains non-letter characters.

    """

    assert re.match(r'^[a-zA-Z]+$', column_title), "Input must contain only letters"

    return sum(pow(26, i) * (ord(c) - ord('A') + 1) for i, c in enumerate(column_title[::-1]))

