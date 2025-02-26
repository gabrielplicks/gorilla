import re



def remove_all_patterns(string: str, pattern: str) -> str:

    """Removes all matches of a given pattern from a given string.



    Args:

        string: The original string.

        pattern: The regular expression pattern to be removed.

    """

    pattern = re.escape(pattern)  # Sanitize the pattern

    pattern = re.compile(pattern, re.IGNORECASE)  # Compile the pattern with case insensitivity

    return pattern.sub('', string)  # Replace all matches with an empty string

