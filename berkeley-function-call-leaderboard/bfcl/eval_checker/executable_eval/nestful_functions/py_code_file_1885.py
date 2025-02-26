from typing import Union



def include_word(word: Union[str, int], include: bool) -> str:

    """Returns a string including the specified word if a boolean parameter is True.

    Otherwise, if the boolean is False, the function returns an empty string.

    Args:

        word: The word to include in the string.

        include: A boolean indicating whether to include the word in the string.

    """

    if include:

        return str(word)

    else:

        return ''

