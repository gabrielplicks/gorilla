import re



def split_and_format_words(string: str) -> list:

    """Splits a given string into a list of words, where each word in the list is separated by a space.

    Capitalizes the first letter of each word, removes all leading and trailing spaces, and removes any consecutive spaces and replaces them with a single space.

    Args:

        string: The input string to be split and formatted.

    Returns:

        A list of formatted words.

    """

    words = string.split()

    formatted_words = [re.sub(r'\s+', ' ', word.strip().capitalize()) for word in words]

    return formatted_words

