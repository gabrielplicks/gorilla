import re



def remove_control_characters(s: str) -> str:

    """Removes all control characters from a string, including newlines.



    Args:

        s: The input string.



    Returns:

        A new string with all control characters removed.

    """

    return re.sub(r'[\x00-\x1F\r]', '', s)

