from typing import List



def remove_trailing_whitespace_and_split(string: str) -> List[str]:

    """Removes all trailing whitespaces from the input string, splits the string into a list of words, uses list comprehension to build the list of words, and raises an exception if the string contains any non-alphabetic characters.



    Args:

        string: The input string.



    Raises:

        ValueError: If the string contains any non-alphabetic characters.

    """

    string = string.lstrip()

    words = string.split()

    words = [word for word in words if word.isalpha()]

    if not all(word.isalpha() for word in words):

        raise ValueError("String contains non-alphabetic characters.")

    return words

