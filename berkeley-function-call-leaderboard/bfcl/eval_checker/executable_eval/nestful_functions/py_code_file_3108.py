from typing import Union



def convert_to_single_line(text: str) -> str:

    """Converts a multi-line string to a single-line string with a space separator.



    Args:

        text: The input multi-line string.



    Returns:

        The converted single-line string.

    """

    lines = text.splitlines()

    return ' '.join(lines)

