import re

import typing



def replace_substring_regex(string: str, regex: str, replacement: str, count: int = 0) -> str:

    """Replaces a substring in a string with a new substring, where the substring to be replaced is a regular expression.



    Args:

        string: The original string.

        regex: The regular expression to be replaced.

        replacement: The new substring to replace with.

        count: The number of replacements to perform. Default is 0, which means all matches will be replaced.

    """

    return re.sub(regex, replacement, string, count=count)

