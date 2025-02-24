import re

from typing import AnyStr



def domain_name_validation(url: AnyStr) -> bool:

    """Validates a URL against the given pattern.



    Args:

        url: The URL to validate.



    Returns:

        True if the URL matches the pattern, False otherwise.

    """

    regex = r'^(https?://)?([a-z0-9]+(\.[a-z0-9]+)+)(:[0-9]+)?(/[a-z0-9\.\-\_]+)*(\?[a-z0-9\.\-\_]+=[a-z0-9\.\-\_]+)?(#?[a-z0-9\.\-\_]+)?$'

    return re.match(regex, url) is not None

