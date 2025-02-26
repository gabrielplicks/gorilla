from typing import List



def split_by_string(s: str, sep: str) -> List[str]:

    """Splits a string `s` into a list of strings based on a `sep` string.

    If `sep` does not exist in `s`, then return a list containing the original string.

    Use ternary operator to return the result list.



    Args:

        s: The input string to be split.

        sep: The separator string.



    Returns:

        A list of strings.

    """

    return [s] if sep not in s else s.split(sep)

