import urllib.parse

from typing import Tuple



def generate_hyperlink(title: str, url: str) -> str:

    """Generates a hyperlink, represented as a string in HTML format, from a given title and URL.

    Args:

        title: The title of the hyperlink.

        url: The URL of the hyperlink.

    Raises:

        Exception: If the URL is not valid.

    """

    try:

        urllib.parse.urlparse(url)

    except Exception as e:

        raise Exception("The URL is not valid")

    return f'<a href="{url}">{title}</a>'

