import re

from typing import AnyStr



def snake_to_kebab(snake_case_string: AnyStr) -> AnyStr:

    """Converts a string from snake_case to kebab-case.

    Args:

        snake_case_string: The string to convert.

    """

    kebab_case_string = re.sub(r'_', '-', snake_case_string.lower())

    return kebab_case_string

