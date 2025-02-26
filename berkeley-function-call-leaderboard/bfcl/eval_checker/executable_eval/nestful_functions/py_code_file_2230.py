from typing import List



def map_to_first_letter(words: List[str]) -> List[str]:

    """Maps a list of strings to another list of strings containing only the first letter of each word.



    Args:

        words: A list of strings.



    Returns:

        A list of strings containing the first letter of each word.

    """

    first_letter_lambda = lambda word: word[0]

    first_letter_dict = {word: first_letter_lambda(word) for word in words}

    first_letters = [letter for letter in first_letter_dict.values()]

    return first_letters

