from typing import List



def split_and_title(title: str) -> str:

    """Formats a title by capitalizing each word and joining them with spaces.



    Args:

        title: The title to be formatted.



    Returns:

        The formatted title.

    """

    words: List[str] = title.split()

    formatted_words: List[str] = [word.capitalize() for word in words]

    return ' '.join(formatted_words)

