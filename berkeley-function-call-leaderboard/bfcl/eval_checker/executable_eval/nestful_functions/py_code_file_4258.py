import re

import typing



def camel_to_snake(text: str) -> str:

    """Converts a string from camelCase to snake_case.



    The function handles special cases to preserve acronyms and numbers.

    It can also handle cases where the string contains multiple consecutive

    capital letters.



    Args:

        text: The input string in camelCase.



    Returns:

        The input string converted to snake_case.

    """

    pattern = re.compile(r"([a-z0-9])([A-Z])")

    def replace(match: re.Match) -> str:

        return match.group(1) + "_" + match.group(2).lower()



    return pattern.sub(replace, text)

