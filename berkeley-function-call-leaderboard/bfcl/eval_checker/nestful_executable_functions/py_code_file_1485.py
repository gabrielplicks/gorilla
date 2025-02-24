import re

from typing import Union



def check_content_type_is_json(headers: str) -> bool:

    """Checks if the Content-Type value in HTTP headers is application/json.



    Args:

        headers: The HTTP headers as a string.



    Returns:

        True if the Content-Type value is application/json, False otherwise.

    """

    content_type_regex = re.compile(r"Content-Type: (?P<content_type>\S+)", re.IGNORECASE)

    match = content_type_regex.search(headers)

    if match:

        content_type: Union[str, None] = match.group("content_type")

        return content_type == "application/json"

    else:

        return False

