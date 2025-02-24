import re



def replace_consecutive_characters(input_string: str) -> str:

    """Replaces consecutive repeated characters in a string with a single instance of the character.



    Args:

        input_string: The input string.



    Returns:

        The modified string.

    """

    return re.sub(r'(.)\1+', r'\1', input_string)

