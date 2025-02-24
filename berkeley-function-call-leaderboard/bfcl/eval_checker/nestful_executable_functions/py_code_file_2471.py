from typing import List, Union



def to_pascal_case(string: str) -> str:

    """Converts a given string to Pascal case.



    Args:

        string: The input string to be converted to Pascal case.



    Returns:

        A new string in Pascal case, where the first letter of each word is capitalized.

    """

    words: List[str] = string.split()

    pascal_words: List[str] = [word.capitalize() for word in words]

    pascal_case: str = ' '.join(pascal_words)

    return pascal_case

