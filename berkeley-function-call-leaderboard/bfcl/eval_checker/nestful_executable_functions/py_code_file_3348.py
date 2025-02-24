from typing import Optional



def remove_longer_suffix(string: str, suffix: str) -> str:

    """Removes the longest possible suffix from a given string that is equal to the given suffix.

    If the suffix is not present, the function returns the original string.

    Returns a string that is a composition of the following format: f'{string} - {suffix}'.

    Args:

        string: The input string.

        suffix: The input suffix.

    """

    if string.endswith(suffix):

        suffix_index = string.rfind(suffix)

        return string[:suffix_index] + ' - ' + suffix

    else:

        return string + ' - ' + suffix

