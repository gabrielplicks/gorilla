import re



def extract_exclamation_marks(string: str) -> int:

    """Extracts all exclamation marks (!) from a given string and returns the count.



    Args:

        string: The input string.



    Returns:

        The count of exclamation marks in the string.

    """

    return sum(1 for _ in re.finditer(r'!', string))

