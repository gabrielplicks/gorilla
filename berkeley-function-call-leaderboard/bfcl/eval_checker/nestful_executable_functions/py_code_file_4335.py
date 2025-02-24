from typing import List



def capitalize_first_letter_of_every_word(string: str) -> str:

    """Capitalizes the first letter of every word in a string.



    Args:

        string: The input string.



    Returns:

        The input string with the first letter of every word capitalized.

    """

    def capitalize_word(word: str) -> str:

        """Capitalizes the first letter of a word.



        Args:

            word: The input word.



        Returns:

            The input word with the first letter capitalized.

        """

        return word.title()



    words = string.split()



    capitalized_words = map(capitalize_word, words)



    return ' '.join(capitalized_words)

