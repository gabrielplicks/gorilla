from typing import Optional



def extract_cookie(header_text: str) -> Optional[str]:

    """Extracts the value of the `Cookie` field from an HTTP header.



    Args:

        header_text: The HTTP header as a string.



    Returns:

        The value of the `Cookie` field if it exists, or None if it does not.

    """

    header_lines = header_text.splitlines()

    for line in header_lines:

        if "Cookie:" in line:

            cookie = line.split(":", 1)[1].strip()

            return cookie

    return None

