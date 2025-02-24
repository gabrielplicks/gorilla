import re



def replace_whitespace(text: str, replacement: str) -> str:

    """Replaces all whitespace characters in a string with a specified replacement string.



    Args:

        text: The input string.

        replacement: The replacement string.

    """

    return re.sub(r"\s+", replacement, text)

