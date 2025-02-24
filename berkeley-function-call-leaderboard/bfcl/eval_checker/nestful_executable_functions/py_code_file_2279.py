import re



def split_string_by_regex(s: str, regex: str) -> list:

    """Splits a string into substrings based on a provided regular expression.



    Args:

        s: The string to be split.

        regex: The regular expression pattern to split the string by.



    Returns:

        A list of substrings.

    """

    return re.split(regex, s)

