from typing import List



def format_array(strings: List[str], separator: str = ',') -> str:

    """Concatenates a list of strings with a given separator.

    If the list has only one item, returns it directly without any changes.

    If the list is empty, returns an empty string.

    Args:

        strings: A list of strings to be concatenated.

        separator: The delimiter to be used between the strings. Defaults to a comma.

    """

    if len(strings) == 0:

        return ''

    if len(strings) == 1:

        return strings[0]

    return separator.join(strings)

