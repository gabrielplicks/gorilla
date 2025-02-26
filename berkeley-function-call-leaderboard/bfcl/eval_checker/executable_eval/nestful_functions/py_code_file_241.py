from typing import List



def get_letters(string: str) -> List[str]:

    """Returns a list containing only the letters of the original string.



    Args:

        string: The input string.



    Returns:

        A list containing only the letters of the original string.

        If the input is not a valid string, an empty list is returned.

    """

    if not isinstance(string, str):

        return []

    letters = []



    for char in string:

        if char.isalpha():

            letters.append(char)

    return letters

