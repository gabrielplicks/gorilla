from typing import AnyStr



def palindrome_checker(word: AnyStr) -> bool:

    """Check if a word is a palindrome.



    Args:

        word: The word to check.



    Returns:

        True if the word is a palindrome, False otherwise.

    """

    for i in range(len(word) // 2):

        if word[i] != word[-i - 1]:

            return False



    return True

