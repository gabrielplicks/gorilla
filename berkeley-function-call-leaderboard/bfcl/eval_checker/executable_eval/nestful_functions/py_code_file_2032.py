from typing import AnyStr



def remove_extra_whitespaces(text: AnyStr) -> AnyStr:

    """Removes unnecessary whitespace from a string.

    Specifically, this function removes any extra whitespace between words,

    as well as any leading or trailing whitespace at the beginning or end of the string.

    Args:

        text: The input string.

    """

    words = text.split()

    return ' '.join(words)

