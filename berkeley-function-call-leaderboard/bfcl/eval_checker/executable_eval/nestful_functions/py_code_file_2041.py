import re

import typing



def replace_abc_with_def(input_string: str) -> str:

    """Replaces all occurrences of the substring "abc" with "def" in the input string.



    Args:

        input_string: The input string to be modified.



    Returns:

        The modified string with all occurrences of "abc" replaced with "def".

    """

    pattern = re.compile(r'abc')

    return pattern.sub('def', input_string)

