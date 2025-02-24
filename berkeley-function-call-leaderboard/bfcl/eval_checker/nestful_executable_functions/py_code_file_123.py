import re



def replace_hyphens(string: str) -> str:

    """Replaces all hyphens with underscores in a given string.



    Args:

        string: The input string.



    Returns:

        A new string with all hyphens replaced with underscores.

    """

    return re.sub(r'-', '_', string)

