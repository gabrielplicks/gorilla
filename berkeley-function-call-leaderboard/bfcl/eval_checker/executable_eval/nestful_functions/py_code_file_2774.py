from typing import AnyStr

import html



def escape_html_characters(text: AnyStr) -> AnyStr:

    """Escapes HTML special characters in a string.



    Args:

        text: The input string.



    Returns:

        The escaped version of the input string.

    """

    return html.escape(text)

