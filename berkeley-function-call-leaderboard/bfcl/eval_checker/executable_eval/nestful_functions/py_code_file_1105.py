import re

from typing import Optional



def generate_url_pattern(url: str) -> Optional[str]:

    """Generates a named URL pattern for a given URL.

    The pattern is in the form of 'app_name:view_name', where 'app_name' is the name of the app and 'view_name' is the name of the view.

    Args:

        url: The URL to generate the pattern for.

    Returns:

        The named URL pattern if the URL matches the expected format, otherwise None.

    """

    pattern = re.compile(r'^http:\/\/example\.com\/(?P<app_name>\w+)\/(?P<view_name>\w+)\/?$')

    match = pattern.match(url)

    if match is not None:

        app_name = match.group('app_name')

        view_name = match.group('view_name')

        return f'{app_name}:{view_name}'

    else:

        return None

