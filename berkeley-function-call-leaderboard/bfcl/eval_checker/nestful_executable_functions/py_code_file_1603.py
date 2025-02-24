from typing import AnyStr



def convert_unicode_to_latin1(unicode_str: AnyStr) -> bytes:

    """Converts a Unicode string to its Latin-1 encoding while stripping leading and trailing whitespaces.



    Args:

        unicode_str: The input Unicode string.



    Raises:

        UnicodeEncodeError: If the input string cannot be encoded with Latin-1.

    """

    filtered_str = unicode_str.strip()

    latin1_bytes = filtered_str.encode("latin-1")

    return latin1_bytes

