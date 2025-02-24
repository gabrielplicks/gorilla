import unicodedata



def remove_diacritics_and_normalize(text: str) -> str:

    """Removes diacritics from a given string and normalizes it for case-insensitive operations.



    Args:

        text: The input string.



    Returns:

        The input string with diacritics removed and normalized for case-insensitive operations.

    """

    normalized_text = unicodedata.normalize('NFD', text)

    without_diacritics = ''.join(c for c in normalized_text if not unicodedata.combining(c))

    return without_diacritics.casefold()

