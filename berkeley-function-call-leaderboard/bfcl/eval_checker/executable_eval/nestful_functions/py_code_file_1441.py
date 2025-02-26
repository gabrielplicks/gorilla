from typing import AnyStr



def remove_capital_letters(string: AnyStr) -> AnyStr:

    """Removes all capital letters from a string.

    Args:

        string: The input string.

    """

    if string.lower() == string:

        return string

    else:

        return string.lower()

