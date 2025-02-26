import string

from typing import List



def clean_words(input_string: str) -> List[str]:

    """

    Removes punctuation from a string and returns a list of words in lower case.

    Args:

        input_string: The input string.

    Returns:

        A list of words.

    """

    try:

        if not isinstance(input_string, str):

            raise TypeError("Invalid input type")

        if not input_string:

            raise ValueError("Input string is empty")

        translator = str.maketrans('', '', string.punctuation)

        cleaned_string = input_string.translate(translator)

        words = cleaned_string.lower().split()



        return words

    except (TypeError, ValueError) as e:

        print(f"Error: {e}")

        return []

