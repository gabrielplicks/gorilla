from typing import Union



def get_sentence(number: int, word: Union[str, int]) -> str:

    """Creates a sentence that includes the word in the appropriate plural form based on the number.



    Args:

        number: The number of items.

        word: The word to be pluralized.



    Returns:

        A sentence that includes the word in the appropriate plural form based on the number.

    """

    if number == 1:

        return f"{number} {word}"

    else:

        return f"{number} {word}s"

