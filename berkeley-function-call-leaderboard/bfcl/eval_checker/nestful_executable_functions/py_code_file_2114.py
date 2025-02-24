import re

import typing



def capitalize_first_letter_of_each_word(s: str) -> str:

    """Capitalizes the first letter of each word in a string.



    Args:

        s: The input string.

    """

    pattern = r"(\s|^)(\w)"

    result = re.sub(pattern, lambda match: match.group(1) + match.group(2).upper(), s)

    return result

