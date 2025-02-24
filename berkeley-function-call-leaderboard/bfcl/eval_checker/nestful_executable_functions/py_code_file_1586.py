import re

import typing



def get_name_from_email(email: str) -> str:

    """Extracts the name from an email address string.



    Args:

        email: A string representing an email address.



    Returns:

        A string with the name and domain parts separated by a single space.

    """

    pattern = r"^(\w[\w\d_\-]*)(@.+)$"

    match = re.search(pattern, email)

    if match:

        name, domain = match.groups()

        return f"{name} {domain}"

    return ""

