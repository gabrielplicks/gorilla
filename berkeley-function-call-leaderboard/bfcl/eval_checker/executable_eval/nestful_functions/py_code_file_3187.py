from typing import List



def split_comma_separated_list(s: str) -> List[str]:

    """Splits a comma-separated list into a list of strings, where each string represents a single item in the list.

    The function removes any leading or trailing whitespace from each item.

    Args:

        s: A string representing a comma-separated list.

    Returns:

        A list of strings, where each string represents a single item in the comma-separated list.

    """

    return [item.strip() for item in s.split(',')]

