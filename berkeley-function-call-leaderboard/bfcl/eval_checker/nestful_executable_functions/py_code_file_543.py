from typing import List



def split_string_by_comma(string: str) -> List[str]:

    """Splits a string into a list of substrings separated by a comma (`,`).



    Args:

        string: The input string.



    Returns:

        A list of substrings.

    """

    if not string or string.isspace():

        return []



    substrings = string.split(",")

    return [substring.replace(",", "") for substring in substrings]

