import re

from typing import Optional



def sanitize(string: str) -> Optional[str]:

    """Sanitizes a string by replacing non-alphanumeric characters with a space,

    removing duplicate spaces, and checking if the string is empty after sanitization.

    Args:

        string: The input string to be sanitized.

    Returns:

        The sanitized string if it is not empty, otherwise an empty string.

    """

    sanitized = re.sub(r"[^0-9a-zA-Z]", " ", string)

    sanitized = " ".join(sanitized.split())

    return sanitized if len(sanitized) > 0 else ""

