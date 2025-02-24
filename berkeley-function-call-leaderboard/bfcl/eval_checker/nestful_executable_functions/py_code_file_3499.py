import re



def replace_apple_with_orange(text: str) -> str:

    """Replaces any occurrence of the substring "apple" with "orange" using a regular expression.

    If the substring "apple" occurs at the end of the string, it is also replaced with "orange".

    Args:

        text: The input string.

    Returns:

        The modified string.

    """

    pattern = re.compile(r'\bapple\b(.*)')  # Match "apple" at the end of the string or followed by any character (.*)

    return pattern.sub(r'orange\1', text)  # Replace "apple" with "orange"

