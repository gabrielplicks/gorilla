from typing import Tuple



def construct_html_tag(tag: str, text: str) -> str:

    """Constructs an HTML element with the given tag and text as content.



    Args:

        tag: The HTML tag to use for the element.

        text: The content to include within the HTML element.



    Returns:

        A string representing the constructed HTML element.

    """

    return f"<{tag}>{text}</{tag}>"

