import keyword



def is_keyword(word: str) -> bool:

    """Checks if a given word is a Python keyword.



    Args:

        word: The word to check.



    Returns:

        True if the word is a Python keyword, False otherwise.

    """

    return keyword.iskeyword(word)

