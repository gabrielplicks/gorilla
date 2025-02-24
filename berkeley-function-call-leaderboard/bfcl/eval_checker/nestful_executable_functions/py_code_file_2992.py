from typing import Union



def starts_with_prefix(word: Union[str, None], prefix: Union[str, None]) -> bool:

    """Checks if the given word starts with the given prefix.



    Args:

        word: The word to check.

        prefix: The prefix to check against.



    Returns:

        True if the word starts with the prefix, False otherwise.

    """

    if not isinstance(word, str) or not isinstance(prefix, str):

        return False

    if len(word) < len(prefix):

        return False

    return word.startswith(prefix)

