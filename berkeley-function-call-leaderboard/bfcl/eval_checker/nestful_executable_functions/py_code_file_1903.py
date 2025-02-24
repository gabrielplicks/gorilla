from typing import List



def split_string_by_delimiter(s: str, delimiter: str) -> List[str]:

    """Splits a string `s` into substrings according to a provided delimiter `delimiter`.



    Args:

        s: The input string to be split.

        delimiter: The delimiter used to split the string.

    """

    substrings = s.split(delimiter)

    return substrings

