from typing import Union



def duplicate_string(string: Union[str, None]) -> Union[str, None]:

    """Duplicates each character in a string.



    Args:

        string: The input string.



    Returns:

        A string where each character is duplicated. If the input string is `None` or not a string,

        the function returns `None`. If the input string is empty, the function returns the empty string.

    """

    if string is None or not isinstance(string, str):

        return None

    if not string:

        return ''



    return ''.join(c * 2 for c in string)

