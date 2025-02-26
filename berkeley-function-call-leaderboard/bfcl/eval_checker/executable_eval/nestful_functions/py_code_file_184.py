from typing import Optional



def convert_to_http_header_name(string: str) -> Optional[str]:

    """Converts a string to uppercase format used in HTTP headers, with words separated by hyphens (-) and without any spaces.



    Args:

        string: The input string.



    Returns:

        The converted string or None if the input is not a string.

    """

    if not isinstance(string, str):

        return None

    return string.upper().replace(" ", "-")

