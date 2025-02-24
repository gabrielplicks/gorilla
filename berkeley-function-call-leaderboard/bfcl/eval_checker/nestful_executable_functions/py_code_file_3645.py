from typing import Union



def skip_first(string: Union[str, bytes]) -> Union[str, bytes]:

    """Returns a new string with all the same characters except the first one.



    Args:

        string: The input string.

    """

    return string[1:]

