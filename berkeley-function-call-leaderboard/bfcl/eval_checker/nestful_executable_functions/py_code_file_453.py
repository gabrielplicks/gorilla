from typing import Set



def validate_keywords(header: str) -> bool:

    """Validates the keywords in a header.



    Args:

        header: The header string to validate.



    Returns:

        A boolean value indicating whether the keywords are valid or not.

    """

    keywords: Set[str] = {'keyword1', 'keyword2', 'keyword3'}

    words: list[str] = header.split()



    for word in words:

        if word not in keywords:

            return False



    return True

