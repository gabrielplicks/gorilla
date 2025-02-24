import re



def replace_is_with_isnt(string: str) -> str:

    """Replaces all occurrences of the word "is" with "isn't" in the input string.



    Args:

        string: The input string to be processed.



    Returns:

        The modified string with all occurrences of the word "is" replaced with "isn't".

    """

    return re.sub(r'\bis\b', "isn't", string)

