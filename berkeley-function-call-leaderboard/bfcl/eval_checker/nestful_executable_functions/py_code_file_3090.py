import re



def strip_digits(text: str) -> str:

    """Removes all digits from a string.



    Args:

        text: The input string.



    Returns:

        The string with all digits removed.

    """

    return re.sub(r"\d", "", text)

