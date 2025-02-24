import re

from typing import AnyStr



def remove_bad_phone_numbers(text: AnyStr) -> AnyStr:

    """Removes phone numbers that are not valid from a given text.



    Args:

        text: The input text containing phone numbers.



    Returns:

        The input text with the bad phone numbers removed.

    """

    regex = re.compile(r"(\d{3})[ -](\d{3})[ -](\d{4})|(\d{3})[.](\d{3})[.](\d{4})|(\d{3})[/](\d{3})[/](\d{4})")

    return re.sub(regex, "", text)

