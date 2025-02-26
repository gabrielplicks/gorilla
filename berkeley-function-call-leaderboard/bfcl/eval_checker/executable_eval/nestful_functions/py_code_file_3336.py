from typing import Literal



ASCII_ZERO: Literal[48] = 48

ASCII_ONE: Literal[49] = 49



def is_binary_word(word: str) -> bool:

    """Checks if a string is a binary word.



    Args:

        word: The string to check.



    Returns:

        True if the string is a binary word, False otherwise.

    """

    assert isinstance(word, str), "Input must be a string"



    for char in word:

        if ord(char) != ASCII_ZERO and ord(char) != ASCII_ONE:

            return False



    return True

