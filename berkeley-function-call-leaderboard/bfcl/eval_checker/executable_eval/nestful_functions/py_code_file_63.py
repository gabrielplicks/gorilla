import re



def replace_first_aces(string: str) -> str:

    """Replaces the first occurrence of the word "ACES" in a string with the word "ACME".



    Args:

        string: The input string.



    Returns:

        The modified string.

    """

    return re.sub("ACES", "ACME", string, count=1)

