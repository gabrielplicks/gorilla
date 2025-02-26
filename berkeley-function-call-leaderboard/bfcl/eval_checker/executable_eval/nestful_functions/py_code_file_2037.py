import re

from typing import Tuple



def match_url_pattern(url: str, pattern: str) -> bool:

    """Checks whether a given URL matches a specific pattern.



    The URL pattern is a string containing a combination of alphanumeric characters, dots (.), hyphens (-), underscores (_), and asterisks (*). The asterisks are placeholders that match any number of characters.



    Args:

        url: The URL to be checked.

        pattern: The URL pattern to be matched.



    Returns:

        True if the URL matches the pattern, False otherwise.

    """

    return re.search(pattern.replace('*', '.*'), url) is not None

