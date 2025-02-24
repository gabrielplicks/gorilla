from typing import List



def strip_list(string_list: List[str]) -> List[str]:

    """Removes any leading or trailing whitespace from each string in a list.



    Args:

        string_list: A list of strings.



    Returns:

        A list of strings with leading or trailing whitespace removed.

    """

    return [string.strip() for string in string_list]

