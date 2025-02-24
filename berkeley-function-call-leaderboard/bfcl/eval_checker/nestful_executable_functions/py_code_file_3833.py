from typing import List, Any, Optional



def list_to_string_list(lst: List[Any]) -> Optional[List[str]]:

    """Converts a list of integers to a list of strings.



    Args:

        lst: The input list.



    Returns:

        A list of strings with each integer represented as a string.

        If the input list is empty, the function returns an empty list.

        The function handles any generic type and returns an optional value.

    """

    if not isinstance(lst, list):

        raise TypeError("Input must be a list.")



    result = []

    for item in lst:

        result.append(str(repr(item)))



    return result

