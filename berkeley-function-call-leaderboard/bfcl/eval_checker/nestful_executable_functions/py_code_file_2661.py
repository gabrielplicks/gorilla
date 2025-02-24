from typing import List, Union



def first_non_empty_string(strings: List[Union[str, None]]) -> Union[str, None]:

    """Returns the first non-empty string from a list. If all strings in the list are empty or if the list is empty, the function returns None.



    Args:

        strings: A list of strings.

    """

    return next((string for string in strings if string), None)

