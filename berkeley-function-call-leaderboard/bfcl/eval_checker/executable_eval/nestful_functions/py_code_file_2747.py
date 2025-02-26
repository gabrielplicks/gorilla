from typing import Union



def default_str(s: Union[str, None]) -> str:

    """Returns the input string if it is not empty, otherwise returns "default".



    Args:

        s: The input string.

    """

    if s != "":

        return s

    else:

        return "default"

