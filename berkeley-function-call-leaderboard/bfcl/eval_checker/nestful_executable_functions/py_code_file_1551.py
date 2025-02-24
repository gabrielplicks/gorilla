import re



def remove_extra_space_between_words(string: str) -> str:

    """Removes extra spaces between words in a string.



    Args:

        string: The input string.



    Returns:

        The modified string with extra spaces removed.

    """

    string = re.sub(r'\s+', ' ', string)

    string = re.sub(r'[\t\n]', ' ', string)

    return string

