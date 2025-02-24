import re



def split_string_by_pattern(string: str, pattern: str) -> list:

    """Splits a string into a list of strings by a given regular expression pattern.

    Args:

        string: The input string.

        pattern: The regular expression pattern to split the string by.

    Returns:

        A list of strings that do not match the specified pattern.

    """

    return [s for s in re.split(pattern, string) if s]

