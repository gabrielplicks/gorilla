import re

from typing import Optional



def extract_phone_number(text: str) -> Optional[str]:

    """Extracts the first phone number from a given string.

    Args:

        text: The input string that may contain one or more phone numbers.

    Returns:

        The first phone number found in the string, or None if no phone number is found.

    """

    pattern = r'(?:(\w\w-)|^)(\d{3}-\d{3}-\d{4})'

    match = re.search(pattern, text)

    if match:

        return match.group(2)

    return None

