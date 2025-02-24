import re

from typing import AnyStr



def is_html_attribute(s: AnyStr) -> bool:

    """Checks whether a string is a valid HTML attribute.



    Args:

        s: The string to check.



    Returns:

        True if the string is a valid HTML attribute, False otherwise.

    """

    pass



def add_display_none(s: AnyStr) -> AnyStr:

    """Converts all occurrences of `style=""` to `style="display: none;"` in a string.



    Args:

        s: The string to convert.



    Returns:

        The converted string.

    """

    return re.sub(r'style=""', r'style="display: none;"', s, flags=re.DOTALL) if is_html_attribute(s) else s

