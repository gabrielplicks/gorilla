import re



def replace_all_pattern(pattern: str, replacement: str, text: str) -> str:

    """Replaces all occurrences of a pattern string with a replacement string in a text string.



    Args:

        pattern: The string to match.

        replacement: The string to replace.

        text: The original text string.



    Returns:

        The modified text string with all occurrences of the pattern replaced with the replacement.

    """

    return re.sub(pattern, replacement, text)

