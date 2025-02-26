from typing import List



def title_version(string: str) -> str:

    """

    Converts a string to its "title version" by capitalizing the first letter of each word and converting the rest of the letters to lowercase.



    Args:

        string: The input string.



    Returns:

        The title version of the input string.

    """

    words: List[str] = string.split()

    title_words: List[str] = [word[0].upper() + word[1:].lower() for word in words]

    title_string: str = " ".join(title_words)

    return title_string

