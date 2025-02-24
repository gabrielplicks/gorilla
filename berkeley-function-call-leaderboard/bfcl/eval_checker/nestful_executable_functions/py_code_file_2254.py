import re



def contains_digit(text: str) -> bool:

    """Determines if a string contains a numeric digit using regular expressions.



    Args:

        text: The input string to check for a numeric digit.



    Returns:

        True if the string contains a digit (0-9), False otherwise.

    """

    return bool(re.search(r"\d", text))

