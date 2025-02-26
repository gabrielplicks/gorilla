from typing import Optional



def strip_line(line: str, replacement: Optional[str] = '') -> str:

    """Strips a line of text of leading and trailing whitespace, or replaces all whitespace with a provided replacement.



    Args:

        line: The line of text to be stripped or modified.

        replacement: The character to replace whitespace with. If not provided, defaults to an empty string.

    """

    if replacement == '':

        return line.strip()

    else:

        return line.replace(' ', replacement)

