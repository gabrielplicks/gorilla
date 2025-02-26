from typing import Tuple



def wrap_text_with_tag(text: str, tag: str) -> str:

    """Wraps the text with an HTML tag.



    Args:

        text: The text to be wrapped.

        tag: The HTML tag to wrap the text with.



    Returns:

        The resulting HTML element.

    """

    return "<{0}>{1}</{0}>".format(tag, text)

