from typing import List



def format_list_strings(strings: List[str]) -> str:

    """Formats a list of strings into a single string, where each string is wrapped in double quotes, followed by a comma and a space. If the original list is empty, returns an empty string instead.



    Args:

        strings: The list of strings to format.



    Returns:

        The formatted string.

    """

    if not strings:

        return ""

    return ", ".join(map(lambda x: '"' + x + '"', strings))

