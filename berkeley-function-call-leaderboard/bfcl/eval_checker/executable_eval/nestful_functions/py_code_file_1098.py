from typing import List



def filter_uppercase_words(text: str) -> List[str]:

    """

    Filters a string and returns a list of words that contain only uppercase letters.



    Args:

        text: The input string to filter.



    Returns:

        A list of words that contain only uppercase letters.

    """

    return [word for word in text.split() if word.isupper()]

