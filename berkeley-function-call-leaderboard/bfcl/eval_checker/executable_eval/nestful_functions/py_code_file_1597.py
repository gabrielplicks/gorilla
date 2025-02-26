from typing import List, Tuple



def get_strings(items: List[Tuple[str, bool]]) -> List[str]:

    """Filters out tuples whose boolean value is False and returns a list of strings.



    Args:

        items: A list of tuples, where each tuple contains a string and a boolean value.



    Returns:

        A list of strings from the tuples whose boolean value is True.

    """

    return [value for value, is_valid in items if is_valid]

