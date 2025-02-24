from typing import Tuple



def format_html_link(text: str, url: str) -> str:

    """Generates a hyperlink using the specified text and URL.

    Args:

        text: The text to display for the hyperlink.

        url: The URL to link to.

    Returns:

        An HTML string with the text inside an `<a>` tag and the URL as the tag's `href` attribute.

    """

    return f'<a href="{url}">{text}</a>'

