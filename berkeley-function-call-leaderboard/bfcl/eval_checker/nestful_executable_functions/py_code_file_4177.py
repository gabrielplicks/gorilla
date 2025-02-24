import re

from typing import Optional



def convert_to_v2_0_0_url(url: str) -> Optional[str]:

    """Converts a URL from version 1.0.0 to version 2.0.0.



    Args:

        url: The input URL.



    Returns:

        The equivalent version 2.0.0 URL if the input URL matches the pattern,

        otherwise None.

    """

    regex = r"https://api.foo.com/v1/(\w+)"

    match = re.search(regex, url)

    if match:

        path = match.group(1)

        return f"https://api.foo.com/v2/0/0/{path}"

    else:

        return None

