from typing import Union



def format_line(line: Union[str, int, float], indent: int = 0) -> str:

    """Formats a line of text by wrapping it in `[]` and adding indentation.



    Args:

        line: The line of text to be formatted.

        indent: The number of spaces to add at the beginning of the line.

    """

    if not isinstance(line, str):

        line = str(line)

    return " " * indent + f"[{line}]"

