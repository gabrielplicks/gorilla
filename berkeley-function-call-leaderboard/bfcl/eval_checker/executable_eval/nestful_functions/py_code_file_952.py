from typing import List, Union



def get_latest_string(string_list: List[Union[str, int, None]]) -> str:

    """Retrieves the latest string from a list of strings, assuming the list is sorted in chronological order.

    If the list is empty or contains only empty strings or non-strings, return an empty string.

    The function should handle any potential errors or exceptions and return "ERROR" in such cases.

    Args:

        string_list: A list of strings, integers, or None.

    """

    if not string_list:

        return ""

    for string in reversed(string_list):

        if string and isinstance(string, str):

            return string

    return "ERROR"

