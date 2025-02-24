import re



def consecutive_characters(s: str) -> bool:

    """Check if a given string contains a sequence of two or more consecutive characters that are the same.



    Args:

        s: The input string.



    Returns:

        True if a sequence of two or more consecutive characters that are the same is found, False otherwise.

    """

    return bool(re.search(r'(.)\1{1,}', s))

