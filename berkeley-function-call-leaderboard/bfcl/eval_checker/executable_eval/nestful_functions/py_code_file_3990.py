import re



def remove_characters_by_regex(string: str, regex: str) -> str:

    """Removes characters from a string that match a given regular expression pattern.



    Args:

        string: The input string.

        regex: The regular expression pattern.



    Raises:

        ValueError: If the input string is not a string.

        ValueError: If the regular expression pattern is not a string.

    """

    if not isinstance(string, str):

        raise ValueError("Input string must be a string.")

    if not isinstance(regex, str):

        raise ValueError("Regular expression pattern must be a string.")

    return re.sub(regex, "", string)

