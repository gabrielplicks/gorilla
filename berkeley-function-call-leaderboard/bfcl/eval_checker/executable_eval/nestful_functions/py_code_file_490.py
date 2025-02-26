import re



def replace_with_regex(string: str, pattern: str, replacement: str) -> str:

    """

    Replaces all occurrences of the pattern in the string with the replacement string.

    The pattern may contain regular expression syntax.

    If an error occurs during compilation or replacement, returns the original string.



    Args:

        string: The input string.

        pattern: The pattern to search for.

        replacement: The string to replace the pattern with.

    """

    try:

        regex = re.compile(pattern)

        return regex.sub(replacement, string)

    except re.error:

        return string

