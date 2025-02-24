from typing import Union



def safe_index(string: str, substring: str, start: int) -> Union[int, None]:

    """Searches for a substring within a string, starting at the specified index.

    Args:

        string: The string to search within.

        substring: The substring to search for.

        start: The index to start the search from.

    Returns:

        The index of the substring if it is found, or -1 if the substring is not found or if an invalid index is provided.

    """

    if not isinstance(string, str) or not isinstance(substring, str) or not isinstance(start, int):

        print("Invalid input type.")

        return -1



    if start < 0 or start >= len(string):

        print("Invalid start index.")

        return -1



    if substring not in string:

        print("Substring not found.")

        return -1

    index = string.find(substring, start)

    return index

