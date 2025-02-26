from typing import List, Union



def get_or_fallback(texts: List[str]) -> Union[str, None]:

    """Returns the first non-empty text from a list of texts or None if there are no valid choices.



    Args:

        texts: A list of text strings.

    """

    return next(filter(lambda text: text, texts), None)

