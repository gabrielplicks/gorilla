import re



def normalize_newlines(text: str) -> str:

    """Normalizes the newlines in the given string so that every newline character is represented by a single character ('\n').



    Args:

        text: The input string.



    Returns:

        The input string with normalized newlines.

    """

    return re.sub(r'\n+', '\n', text, flags=re.MULTILINE)

