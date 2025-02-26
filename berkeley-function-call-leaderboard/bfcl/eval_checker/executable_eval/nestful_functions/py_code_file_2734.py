from typing import List



def list_to_single_line_string(lst: List[str]) -> str:

    """

    Converts a Python list to a single line string.

    Example: [1, 2, 3] -> "[1, 2, 3]"

    Args:

        lst: The list to be converted.

    """

    return "[" + ", ".join([str(el) for el in lst]) + "]"

