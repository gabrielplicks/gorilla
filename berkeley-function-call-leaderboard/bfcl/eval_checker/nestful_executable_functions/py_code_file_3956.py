import re

import typing



def count_kmp(s: str) -> int:

    """Counts the number of non-overlapping instances of the string "KMP" in the input string `s`.



    Args:

        s: The input string to search for instances of "KMP".



    Returns:

        The count of non-overlapping instances of "KMP" in `s`.

    """

    return len(re.findall('KMP', s))

