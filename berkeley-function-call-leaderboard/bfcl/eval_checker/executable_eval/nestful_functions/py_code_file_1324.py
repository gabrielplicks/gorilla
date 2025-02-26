def get_suffix_index(string: str, suffix: str) -> int:

    """Returns the index of the suffix in the string.



    Args:

        string: The original string.

        suffix: The suffix to search for.



    Returns:

        The index of the suffix in the string, or -1 if the suffix is not found.

    """

    if suffix is None:

        return -1

    else:

        if string.endswith(suffix):

            return string.index(suffix)

        else:

            return -1

