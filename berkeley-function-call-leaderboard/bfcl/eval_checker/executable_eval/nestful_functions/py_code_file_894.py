import re

import typing



def is_html_tag(tag: str) -> bool:

    """Checks if a string matches the regex pattern for an HTML tag or a closing tag.



    Args:

        tag: The string to check.



    Returns:

        True if the string matches the pattern, False otherwise.

    """

    html_pattern = r"<[a-zA-Z0-9]+>"

    closing_tag_pattern = r"</[a-zA-Z0-9]+>"



    return bool(re.match(html_pattern, tag)) or bool(re.match(closing_tag_pattern, tag))

