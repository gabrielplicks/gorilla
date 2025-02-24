from typing import List



def filter_hello(strings: List[str]) -> List[str]:

    """Returns a new list with only the strings that match the exact pattern "hello".



    Args:

        strings: The list of strings to filter.



    Returns:

        A new list with only the strings that match the exact pattern "hello".

    """

    return [string for string in strings if string == "hello"]

