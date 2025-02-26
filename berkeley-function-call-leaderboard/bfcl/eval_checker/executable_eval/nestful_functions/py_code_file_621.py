from typing import List



def split_and_capitalize(str: str, separator: str) -> List[str]:

    """Splits a string based on a separator and capitalizes the first letter of each word.



    Args:

        str: The input string.

        separator: The separator used to split the string.



    Returns:

        A list of capitalized words.

    """

    words = str.split(separator)

    capitalized_words = [word.capitalize() for word in words]

    return capitalized_words

