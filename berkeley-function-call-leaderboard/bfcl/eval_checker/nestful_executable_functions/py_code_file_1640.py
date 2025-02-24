import re



def remove_special_characters(string: str) -> str:

    """Removes special characters, leading and trailing spaces, and multiple spaces from a string.



    Args:

        string: The input string.



    Returns:

        The modified string.

    """

    string = string.strip()

    string = re.sub(r'\s+', ' ', string)

    string = re.sub(r'[^\w\s]', '', string)



    return string

