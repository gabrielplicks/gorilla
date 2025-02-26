from typing import Optional



def find_string(source: str, search: str) -> Optional[int]:

    """Returns the first index where the search string occurs in the source string.

    If the search string is not found, returns -1 as a default value.

    Args:

        source: The source string to search in.

        search: The search string to find in the source string.

    """

    return source.find(search)

