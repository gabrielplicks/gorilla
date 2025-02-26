import string



def validate_unique_words(text: str) -> bool:

    """Validates if all of the words in the given string are unique.



    Args:

        text: The input string.



    Returns:

        A boolean value indicating whether all the words in the string are unique.

    """

    words = text.split(' ')

    words = [word.strip(string.punctuation) for word in words]

    words = [word.lower() for word in words]

    return len(words) == len(set(words))

