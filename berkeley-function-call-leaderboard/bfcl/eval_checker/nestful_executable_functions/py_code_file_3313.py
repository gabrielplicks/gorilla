from typing import List



def split_at_str(s: str, sep: str) -> List[str]:

    """Splits a string `s` at a separator `sep` and returns a list of substrings.

    If `sep` is not found in `s`, returns a single-element list containing `s`.

    Args:

        s: The string to be split.

        sep: The separator string.

    """

    return s.split(sep)

