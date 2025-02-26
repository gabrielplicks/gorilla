from typing import Tuple



def generate_html_anchor(url: str, text: str) -> str:

    """Generates an HTML anchor tag with the given URL and text.



    Args:

        url: The URL to be used in the anchor tag.

        text: The text to be displayed in the anchor tag.



    Returns:

        A string containing the HTML code for the anchor tag.

    """

    return '<a href="{}">{}</a>'.format(url, text)

