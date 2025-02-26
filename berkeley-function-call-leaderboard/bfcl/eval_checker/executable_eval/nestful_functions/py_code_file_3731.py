import re



def replace_all_non_alpha_numeric(s: str) -> str:

    """Replaces all non-alphanumeric characters in a string with a space.



    Args:

        s: The input string.



    Returns:

        The modified string with non-alphanumeric characters replaced with a space.

    """

    return re.sub(r'[^0-9a-zA-Z]+', ' ', s)

