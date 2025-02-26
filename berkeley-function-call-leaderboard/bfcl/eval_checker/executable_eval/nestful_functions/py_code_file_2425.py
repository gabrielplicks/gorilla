def last_occurrence(string: str, substring: str) -> int:

    """Returns the last occurrence of a substring in a given string.

    Args:

        string: The string in which to search for the substring.

        substring: The substring to search for.

    """

    return string.rfind(substring)

