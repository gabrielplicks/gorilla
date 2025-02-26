def case_insensitive_equality(str1: str, str2: str) -> bool:

    """

    Returns True if the two input strings are equal, ignoring case.



    Args:

        str1: The first string to compare.

        str2: The second string to compare.



    Returns:

        True if the strings are equal, ignoring case. False otherwise.

    """

    return str1.lower() == str2.lower()

