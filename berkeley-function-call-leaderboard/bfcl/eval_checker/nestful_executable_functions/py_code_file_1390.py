from typing import List



def camelCase(text: str) -> str:

    """Converts a phrase to Camel Case.



    Args:

        text: The input phrase to convert to Camel Case.



    Returns:

        The Camel Case string.

    """

    words: List[str] = text.split()

    camelCaseWords: List[str] = [word[0].upper() + word[1:].lower() for word in words]

    camelCaseString: str = ''.join(camelCaseWords)

    return camelCaseString

