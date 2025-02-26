from typing import Optional



def compare_insensitively(str1: Optional[str], str2: Optional[str]) -> bool:

    """Compares two strings insensitively, but only if both strings are non-empty.



    Args:

        str1: The first string to compare.

        str2: The second string to compare.



    Returns:

        True if the strings are equal, False otherwise.

    """

    str1_lower = str1.casefold() if str1 else ""

    str2_lower = str2.casefold() if str2 else ""

    str1_stripped = str1_lower.strip()

    str2_stripped = str2_lower.strip()

    if str1_stripped and str2_stripped:

        return str1_stripped == str2_stripped

    else:

        return False

