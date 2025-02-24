from typing import Union



def convert_codepoint_to_usv(codepoint: Union[int, str]) -> str:

    """Converts a Unicode code point to a USV string.

    Args:

        codepoint: The Unicode code point to convert.

    """

    if not isinstance(codepoint, int):

        codepoint = int(codepoint, 16)

    if codepoint < 0 or codepoint > 0x10FFFF:

        return ""

    if 0xD800 <= codepoint <= 0xDFFF:

        return ""

    return chr(codepoint)

