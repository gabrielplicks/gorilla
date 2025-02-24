from typing import List



def get_strings_starting_with(strings: List[str], prefix: str) -> List[str]:

    """

    Returns a list of strings that start with the specified prefix.



    Args:

        strings: A list of strings.

        prefix: The prefix to filter the strings with.

    """

    return [s for s in strings if s.startswith(prefix)]

