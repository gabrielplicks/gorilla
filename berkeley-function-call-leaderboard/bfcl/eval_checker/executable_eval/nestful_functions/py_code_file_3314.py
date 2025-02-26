from typing import TextIO



def word_count(text_file: str, word: str) -> int:

    """Counts the number of times a word appears in a text file.



    The word is case-insensitive, and the function ignores any leading or trailing white spaces.

    The function uses a scripting language for file parsing and line-by-line reading.



    Args:

        text_file: The path to the text file.

        word: The word to count.



    Returns:

        The number of times the word appears in the text file.

    """

    with open(text_file, 'r') as file:

        count = 0

        temp_word = ''

        for line in file:

            for char in line:

                if char.isalnum():

                    temp_word += char

                else:

                    if temp_word.lower() == word.lower():

                        count += 1

                    temp_word = ''



        return count

