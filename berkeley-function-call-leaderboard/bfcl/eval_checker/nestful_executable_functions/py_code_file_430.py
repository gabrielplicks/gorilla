import re

from typing import Optional



def swap_tags(html_string: str) -> Optional[str]:

    """Swaps the opening and closing tags in an HTML string.



    Args:

        html_string: The HTML string containing tags and attributes.



    Returns:

        The modified HTML string with swapped opening and closing tags.

    """

    match = re.match(r'<(?P<tag>\w+).*?>', html_string)

    if match:

        tag = match.group('tag')

        opening_tag = f"<{tag}>"

        closing_tag = f"</{tag}>"

        return html_string.replace(opening_tag, closing_tag).replace(closing_tag, opening_tag)

    else:

        return html_string

