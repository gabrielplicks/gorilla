import re



def check_if_contains_digit(word: str) -> bool:

    """Checks if a word contains a digit.

    Args:

        word: The word to check.

    Returns:

        True if the word contains a digit, False otherwise.

    """

    return bool(re.search(r'\d', word))



def extract_words_with_digits(s: str) -> list[str]:

    """Extracts all words with digits from a string.

    Args:

        s: The string to extract words from.

    Returns:

        A list of words that contain at least one digit.

    """

    words = s.split()

    return [word for word in words if check_if_contains_digit(word)]

