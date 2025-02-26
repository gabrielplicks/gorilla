from typing import Optional



def select_branch(string: str) -> bool:

    """Determines if a string contains the substring "th".



    Args:

        string: The input string.



    Returns:

        True if the string contains "th", False otherwise.

    """

    if not string:

        return False

    if "th" in string:

        return True

    return select_branch(string[1:])

