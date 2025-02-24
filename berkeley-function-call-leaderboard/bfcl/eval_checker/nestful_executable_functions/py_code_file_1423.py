import re

import typing



def replace_regex_list(s: str, regex_list: typing.List[str]) -> str:

    """Replaces all occurrences of the regular expressions in regex_list with a space character in s.



    Args:

        s: The input string.

        regex_list: A list of regular expressions.

    """

    for regex in regex_list:

        s = re.sub(regex, ' ', s)

    return s

